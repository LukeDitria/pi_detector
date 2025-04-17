#!/usr/bin/env python3
import struct
import smbus
import time
from datetime import datetime, timedelta
import csv
import os
import subprocess
import logging
import json

from astral import LocationInfo
from astral.sun import sun
from astral.geocoder import database, lookup

import pytz
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Raspi Battery monitor")
    parser.add_argument("--log_file_path", type=str, default="battery_logs.csv",
                        help="Directory to save detection results")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("--device_location", type=str, default="Melbourne",
                        help="Directory to save detection results")
    parser.add_argument("--log_rate_min", type=int, default=5,
                        help="Logging every (x) minutes (default: 5)")
    parser.add_argument("--shutdown_offset", type=int, default=0,
                        help="Offset (hours) to add from shutdown time pos/neg")
    parser.add_argument("--wakeup_offset", type=int, default=0,
                        help="Offset (hours) to add from wakeup time pos/neg")
    parser.add_argument("--project_id", type=str,
                        help="Google Cloud project ID")
    parser.add_argument("--firestore_collection", type=str, default="CameraBox",
                        help="This project name to be stored on Firestore")
    parser.add_argument("--operation_time", type=str,
                        help="When the device will operate: day, night, all", default='all',
                        choices=["day", "night", "all"])
    parser.add_argument("--low_battery_voltage", type=float, default=3.2,
                        help="Battery Voltage to shutdown at")
    parser.add_argument("--log_remote", action='store_true', help="Log to remote store")
    parser.add_argument("--suptronics_ups", action='store_true',
                        help="Is a X1202 or X1206 UPS being used?")

    return parser.parse_args()


class BatteryMonitor:
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

        self.args = parse_arguments()

        # Load config file if provided and override CLI args
        if self.args.config_file:
            try:
                with open(self.args.config_file, 'r') as f:
                    config = json.load(f)

                # Override CLI args with JSON config values
                for key, value in config.items():
                    if hasattr(self.args, key):
                        setattr(self.args, key, value)

                logging.info(f"Loaded configuration from {self.args.config_file}")
            except Exception as e:
                logging.info(f"Error loading config file: {e}")
                logging.info("Using command line arguments instead")

        self.battery_monitor_available = True

        # Replace with your location coordinates and timezone
        try:
            self.location = lookup(self.args.device_location, database())
        except KeyError:
            logging.info("Location not found")

        self.timezone = pytz.timezone(self.location.timezone)
        self.ensure_log_file_exists()

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

        # Try to initialize the battery monitor
        try:
            self.bus = smbus.SMBus(1)
            self.address = 0x36
            # Test if we can read from the device
            self.read_voltage()
        except Exception as e:
            logging.info(f"Battery monitor initialization failed: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False

        if self.args.log_remote:
            from firestore_logger import FirestoreLogger
            logging.info(f"Firestore remote logging")
            try:
                self.fire_logger = FirestoreLogger(project_id=self.args.project_id,
                                                   firestore_collection=self.args.firestore_collection)
                logging.info(f"Firestore logging initialized")
            except Exception as e:
                logging.info(f"Firestore initialization failed: {e}")
                logging.info("Continuing without remote logging")
                self.fire_logger = None
        else:
            self.fire_logger = None

        if self.args.suptronics_ups:
            import gpiozero
            self.charge_pin = gpiozero.DigitalOutputDevice(16)
            self.charge_pin.off()

    def create_log_dict(self, battery_voltage, battery_capacity, status):
        doc_data = {
            "type": "battery_status",
            "timestamp": time.time(),
            "battery_voltage": battery_voltage,
            "battery_capacity": battery_capacity,
            "status": status
        }
        return doc_data

    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist"""
        if not os.path.exists(self.args.log_file_path):
            with open(self.args.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Voltage', 'Capacity', 'Shutdown_Reason'])

    def read_voltage(self):
        """Read battery voltage"""
        if not self.battery_monitor_available:
            return None

        try:
            read = self.bus.read_word_data(self.address, 2)
            swapped = struct.unpack("<H", struct.pack(">H", read))[0]
            voltage = swapped * 1.25 / 1000 / 16
            return voltage
        except Exception as e:
            logging.info(f"Error reading voltage: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False
            return None

    def read_capacity(self):
        """Read battery capacity"""
        if not self.battery_monitor_available:
            return None

        try:
            read = self.bus.read_word_data(self.address, 4)
            swapped = struct.unpack("<H", struct.pack(">H", read))[0]
            capacity = swapped / 256
            return capacity
        except Exception as e:
            logging.info(f"Error reading capacity: {e}")
            logging.info("Continuing without battery monitoring")
            self.battery_monitor_available = False
            return None

    def log_data(self, shutdown_reason=None):
        """Log the current battery voltage to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        voltage = self.read_voltage()
        capacity = self.read_capacity()

        # Always try to log to local file first
        try:
            with open(self.args.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if voltage is not None:
                    writer.writerow([timestamp, f"{voltage:.2f}", f"{capacity:.2f}", shutdown_reason or "on"])
                    logging.info(f"Logged: Time: {timestamp}, Voltage: {voltage:.2f}V, Capacity: {capacity:.2f}%")
                else:
                    writer.writerow([timestamp, "N/A", shutdown_reason or "on (no battery data)"])
                    logging.info(f"Logged: Time: {timestamp}, Voltage: N/A (battery monitor unavailable)")
        except Exception as e:
            logging.info(f"Error logging to file: {e}")

        # Then try to log to Firestore if enabled
        if self.args.log_remote and self.fire_logger:
            try:
                if voltage is not None:
                    doc_dict = self.create_log_dict(voltage, capacity, shutdown_reason or "on")
                else:
                    doc_dict = self.create_log_dict(None, None, shutdown_reason or "on (no battery data)")

                self.fire_logger.log_data_to_firestore(doc_dict,
                                                       doc_type="battery",
                                                       timestamp=timestamp)
            except Exception as e:
                logging.info(f"Error during remote logging: {e}")

    def perform_shutdown(self, reason):
        """Perform system shutdown with proper logging and error handling"""
        logging.info(f"Initiating shutdown due to: {reason}")
        logging.info("Logging final reading before shutdown...")
        self.log_data(shutdown_reason=reason)

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

    def check_shutdown_time(self):
        """Check if current time is after the shutdown time"""
        try:
            now = datetime.now(self.timezone)
            return now > self.shutdown_time
        except Exception as e:
            logging.info(f"Error checking sunset time: {e}")
            return False

    def set_low_power_wakeup(self):
        """Set the wakeup time after a low battery shutdown"""
        now = datetime.now(self.timezone)
        hour_later = now + timedelta(hours=1)

        # Set startup time to next wakeup time if hour_later alarm would be after next shutdown
        # If it's operating during the night just shutdown, because there is no sun!
        if hour_later > self.shutdown_time or self.args.operation_time == "night":
            self.set_alarm(self.startup_time)
        else:
            self.set_alarm(hour_later)

    def set_alarm(self, alarm_time):
        try:
            # Clear any existing alarm
            subprocess.run(["sudo", "sh", "-c", "echo 0 > /sys/class/rtc/rtc0/wakealarm"], check=True)
            # Set new alarm for sunrise
            subprocess.run(["sudo", "sh", "-c", f"echo {int(alarm_time.timestamp())} > /sys/class/rtc/rtc0/wakealarm"],
                           check=True)

            logging.info(f"Wake alarm set for {alarm_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        except subprocess.CalledProcessError as e:
            logging.info(f"Error setting wake alarm: {e}")
            return False

    def check_shutdown_condition(self):
        """Check if it's after sunset or if battery voltage is critically low"""
        # Shutdown if after sunset
        if self.sleep_wake:
            if self.check_shutdown_time():
                self.set_alarm(self.startup_time)
                self.perform_shutdown("Shutdown! After sunset!")

        # Only check battery voltage if monitor is available
        if not self.battery_monitor_available:
            return

        current_voltage = self.read_voltage()
        if current_voltage is None:
            return

        # Shutdown if battery is critically low
        if current_voltage < self.args.low_battery_voltage:
            self.set_low_power_wakeup()
            self.perform_shutdown("Shutdown! Low battery!")

    def run_monitor(self):
        logging.info("Wait for startup!")
        time.sleep(20)
        try:
            logging.info(f"Battery monitoring started. Logging to {self.args.log_file_path}")

            # Print status information
            status_messages = []
            if not self.battery_monitor_available:
                status_messages.append("battery monitoring disabled")
            if self.args.log_remote and not self.fire_logger:
                status_messages.append("remote logging disabled")

            if status_messages:
                logging.info(f"Running with {' and '.join(status_messages)}")

            while True:
                self.log_data()
                self.check_shutdown_condition()

                # Wait for specified minutes
                time.sleep(self.args.log_rate_min * 60)

        except KeyboardInterrupt:
            logging.info("\nMonitoring stopped by user")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            sys.exit(1)


def main():
    monitor = BatteryMonitor()
    monitor.run_monitor()


if __name__ == "__main__":
    main()