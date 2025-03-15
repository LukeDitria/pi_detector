#!/usr/bin/env python3
import struct
import smbus
import time
from datetime import datetime
import csv
import os
import subprocess
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
    parser.add_argument("--device_location", type=str, default="Melbourne",
                        help="Directory to save detection results")
    parser.add_argument("--log_rate_min", type=int, default=5,
                        help="Logging every (x) minutes (default: 5)")
    parser.add_argument("--project_id", type=str,
                        help="Google Cloud project ID")
    parser.add_argument("--low_battery_voltage", type=float, default=3.2,
                        help="Battery Voltage to shutdown at")
    parser.add_argument("--log_remote", action='store_true', help="Log to remote store")
    return parser.parse_args()


class BatteryMonitor:
    def __init__(self):
        self.args = parse_arguments()

        self.battery_monitor_available = True
        self.firestore_available = False

        # Try to initialize the battery monitor
        try:
            self.bus = smbus.SMBus(1)
            self.address = 0x36
            # Test if we can read from the device
            self.read_voltage()
        except Exception as e:
            print(f"Battery monitor initialization failed: {e}")
            print("Continuing without battery monitoring")
            self.battery_monitor_available = False

        # Replace with your location coordinates and timezone
        try:
            self.location = lookup(self.args.device_location, database())
        except KeyError:
            print("Location not found")

        self.timezone = pytz.timezone(self.location.timezone)
        self.ensure_log_file_exists()

        # Initialize Firestore with error handling
        if self.args.log_remote:
            from google.cloud import firestore
            from google.cloud import storage

            try:
                if self.args.project_id is not None:
                    self.db = firestore.Client(project=self.args.project_id)
                    self.storage_client = storage.Client(project=self.args.project_id)
                    # Test the connection by attempting a simple operation
                    self.db.collection('battery').document('test').get()
                    print("Firestore connection established successfully")
                    self.firestore_available = True
                else:
                    print("Firestore Project ID not Provided!")
                    print("Firestore NOT LOGGING!")
                    self.firestore_available = False
            except Exception as e:
                print(f"Firestore initialization failed: {e}")
                print("Continuing without remote logging")
                self.firestore_available = False

    def log_battery_to_firestore(self, battery_voltage, status):
        """Log detection results to Firestore."""
        if not self.args.log_remote or not self.firestore_available:
            return

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            doc_ref = self.db.collection('battery').document(timestamp)

            doc_data = {
                "battery_voltage": battery_voltage,
                "status": status
            }

            doc_ref.set(doc_data)
            print(f"Successfully logged to Firestore: {status}")
        except Exception as e:
            print(f"Error logging to Firestore: {e}")

    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist"""
        if not os.path.exists(self.args.log_file_path):
            with open(self.args.log_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Voltage', 'Shutdown_Reason'])

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
            print(f"Error reading voltage: {e}")
            print("Continuing without battery monitoring")
            self.battery_monitor_available = False
            return None

    def log_data(self, shutdown_reason=None):
        """Log the current battery voltage to CSV file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        voltage = self.read_voltage()

        # Always try to log to local file first
        try:
            with open(self.args.log_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if voltage is not None:
                    writer.writerow([timestamp, f"{voltage:.2f}", shutdown_reason or "on"])
                    print(f"Logged: Time: {timestamp}, Voltage: {voltage:.2f}V")
                else:
                    writer.writerow([timestamp, "N/A", shutdown_reason or "on (no battery data)"])
                    print(f"Logged: Time: {timestamp}, Voltage: N/A (battery monitor unavailable)")
        except Exception as e:
            print(f"Error logging to file: {e}")

        # Then try to log to Firestore if enabled
        if self.args.log_remote and self.firestore_available:
            try:
                if voltage is not None:
                    self.log_battery_to_firestore(voltage, shutdown_reason or "on")
                else:
                    self.log_battery_to_firestore(None, shutdown_reason or "on (no battery data)")
            except Exception as e:
                print(f"Error during remote logging: {e}")

    def is_after_sunset(self):
        """Check if current time is after sunset"""
        try:
            now = datetime.now(self.timezone)
            s = sun(self.location.observer, date=now)
            return now > s['sunset']
        except Exception as e:
            print(f"Error checking sunset time: {e}")
            return False

    def perform_shutdown(self, reason):
        """Perform system shutdown with proper logging and error handling"""
        print(f"Initiating shutdown due to: {reason}")
        print("Logging final reading before shutdown...")
        self.log_data(shutdown_reason=reason)

        # Sync filesystem to ensure logs are written
        try:
            subprocess.run(['sync'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error syncing filesystem: {e}")

        print("Shutting down in 5 seconds...")
        time.sleep(5)

        try:
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to shutdown: {e}")
            sys.exit(1)

    def set_sunrise_wakeup(self):
        """Set the wakeup time to be at sunrise the next morning"""
        now = datetime.now(self.timezone)
        tomorrow = now + timedelta(days=1)
        s = sun(self.location.observer, date=tomorrow)
        sunrise = s['sunrise']

        # Convert to Unix timestamp for wake-alarm
        sunrise_timestamp = int(sunrise.timestamp())

        try:
            # Clear any existing alarm
            subprocess.run(["sudo", "sh", "-c", "echo 0 > /sys/class/rtc/rtc0/wakealarm"], check=True)
            # Set new alarm for sunrise
            subprocess.run(["sudo", "sh", "-c", f"echo {sunrise_timestamp} > /sys/class/rtc/rtc0/wakealarm"],
                           check=True)

            print(f"Wake alarm set for sunrise at {sunrise.strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error setting wakealarm: {e}")
            return False

    def check_shutdown_condition(self):
        """Check if it's after sunset or if battery voltage is critically low"""
        # Shutdown if after sunset
        if self.is_after_sunset():
            self.perform_shutdown("Shutdown! After sunset!")

        # Only check battery voltage if monitor is available
        if not self.battery_monitor_available:
            return

        current_voltage = self.read_voltage()
        if current_voltage is None:
            return

        # Shutdown if battery is critically low
        if current_voltage < self.args.low_battery_voltage:
            self.perform_shutdown("Shutdown! Low battery!")

    def run_monitor(self):
        try:
            time.sleep(5)

            print(f"Battery monitoring started. Logging to {self.args.log_file_path}")

            # Print status information
            status_messages = []
            if not self.battery_monitor_available:
                status_messages.append("battery monitoring disabled")
            if self.args.log_remote and not self.firestore_available:
                status_messages.append("remote logging disabled")

            if status_messages:
                print(f"Running with {' and '.join(status_messages)}")

            while True:
                self.log_data()
                self.check_shutdown_condition()

                # Wait for specified minutes
                time.sleep(self.args.log_rate_min * 60)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)


def main():
    monitor = BatteryMonitor()
    monitor.run_monitor()


if __name__ == "__main__":
    main()