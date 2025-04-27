#!/usr/bin/env python3
import struct
import smbus
import time
from datetime import datetime, timedelta
import os
import subprocess
import logging
from typing import Union, Generator, List, Optional

from astral import LocationInfo
from astral.sun import sun
from astral.geocoder import database, lookup

import pytz
import sys
import get_args
from battery_monitors.suptron_ups import SupTronicsBatteryMonitor

class DeviceMonitor:
    def __init__(self):
        # Set up logging to stdout (systemd will handle redirection)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),  # Logs go to stdout (captured by systemd)
                logging.StreamHandler(sys.stderr)  # Warnings and errors go to stderr
            ]
        )
        logging.info("Capture Box Awake!")

        # First parse command line arguments with all defaults
        self.args = get_args.parse_arguments()

        battery_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args.log_file_path)
        self.battery_monitor = SupTronicsBatteryMonitor(log_file_path=battery_log_path,
                                                        low_battery_voltage=self.args.low_battery_voltage,
                                                        log_remote=self.args.log_remote,
                                                        project_id=self.args.project_id,
                                                        device_name=self.args.device_name)

        # Replace with your location coordinates and timezone
        try:
            self.location = lookup(self.args.device_location, database())
        except KeyError:
            logging.info("Location not found")

        self.timezone = pytz.timezone(self.location.timezone)
        self.sleep_wake = False
        now = datetime.now(self.timezone)
        if self.args.operation_time == "day":
            self.sleep_wake = True
            s = sun(self.location.observer, date=now)
            self.shutdown_time = s["sunset"]
            next_day = now + timedelta(days=1)
            next_s = sun(self.location.observer, date=next_day)
            self.startup_time = next_s["sunrise"]

        elif self.args.operation_time == "night":
            self.sleep_wake = True
            # If the device has woken up after midnight
            if now.hour < 12:
                days_delta = 0
            else:
                days_delta = 1

            next_day = datetime.now(self.timezone) + timedelta(days=days_delta)
            next_s = sun(self.location.observer, date=next_day)
            self.shutdown_time = next_s["sunrise"]
            self.startup_time = next_s["sunset"]

        elif self.args.operation_time == "all":
            self.shutdown_time = None
            self.startup_time = None
        else:
            logging.error("operation_time should be day/night/all")

        if self.sleep_wake:
            self.shutdown_time += timedelta(hours=self.args.shutdown_offset)
            self.startup_time += timedelta(hours=self.args.wakeup_offset)

            logging.info(f"Shutdown Time {self.shutdown_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Startup Time {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            logging.info(f"ALWAYS ON!")

    def perform_shutdown(self, reason: str):
        """Perform system shutdown with proper logging and error handling"""
        logging.info(f"Initiating shutdown due to: {reason}")
        logging.info("Logging final reading before shutdown...")
        timestamp = datetime.now(self.timezone)
        self.battery_monitor.log_data(timestamp, shutdown_reason=reason)

        # Sync filesystem to ensure logs are written
        try:
            subprocess.run(['sync'], check=True)
        except subprocess.CalledProcessError as e:
            logging.info(f"Error syncing filesystem: {e}")

        logging.info("Shutting down in 5 seconds...")
        time.sleep(5)

        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as e:
            logging.info(f"Failed to shutdown: {e}")
            sys.exit(1)

    def check_shutdown_time(self) -> bool:
        """Check if current time is after the shutdown time"""
        try:
            now = datetime.now(self.timezone)
            return now > self.shutdown_time
        except Exception as e:
            logging.info(f"Error checking sunset time: {e}")
            return False

    def set_low_power_wakeup(self):
        """Set the wakeup time after a low battery shutdown"""
        try:
            now = datetime.now(self.timezone)
            hour_later = now + timedelta(hours=1)

            # Set startup time to next wakeup time if hour_later alarm would be after next shutdown
            # If it's operating during the night just shutdown, because there is no sun!
            if self.args.operation_time == "day" and hour_later > self.shutdown_time:
                self.set_alarm(self.startup_time)
            else:
                self.set_alarm(hour_later)
        except Exception as e:
            logging.info(f"Error setting low-power wakeup: {e}")

    def set_alarm(self, alarm_time: datetime) -> None:
        try:
            # Clear any existing alarm
            subprocess.run(["sudo", "sh", "-c", "echo 0 > /sys/class/rtc/rtc0/wakealarm"], check=True)
            # Set new alarm for sunrise
            subprocess.run(["sudo", "sh", "-c", f"echo {int(alarm_time.timestamp())} > /sys/class/rtc/rtc0/wakealarm"],
                           check=True)

            logging.info(f"Wake alarm set for {alarm_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except subprocess.CalledProcessError as e:
            logging.info(f"Error setting wake alarm: {e}")

    def check_shutdown_condition(self) -> None:
        """Check if it's after sunset or if battery voltage is critically low"""
        # Shutdown if after sunset
        if self.sleep_wake:
            if self.check_shutdown_time():
                self.set_alarm(self.startup_time)
                self.perform_shutdown("Shutdown! After sunset!")

        # Only check battery voltage if monitor is available
        if self.battery_monitor:
            # Shutdown if battery is critically low
            if self.battery_monitor.is_battery_low():
                self.set_low_power_wakeup()
                self.perform_shutdown("Shutdown! Low battery!")

    def run_monitor(self):
        logging.info("Wait for startup!")
        time.sleep(self.args.start_delay)
        try:
            while True:
                timestamp = datetime.now(self.timezone)
                self.battery_monitor.log_data(timestamp)
                self.check_shutdown_condition()

                # Wait for specified minutes
                time.sleep(self.args.log_rate_min * 60)

        except KeyboardInterrupt:
            logging.info("\nMonitoring stopped by user")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            sys.exit(1)


def main():
    monitor = DeviceMonitor()
    monitor.run_monitor()


if __name__ == "__main__":
    main()