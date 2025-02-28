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
from google.cloud import firestore
from google.cloud import storage
import pytz
import sys
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Raspi Battery monitor")
    parser.add_argument("--log_file_path", type=str, default="battery_logs.csv",
                        help="Directory to save detection results")
    parser.add_argument("--log_rate_min", type=int, default=5,
                        help="Logging every (x) minutes (default: 5)")
    parser.add_argument("--project_id", type=str, required=True,
                        help="Google Cloud project ID")
    parser.add_argument("--log_remote", action='store_true', help="Log to remote store")
    return parser.parse_args()


class BatteryMonitor:
    def __init__(self, log_file_path, project_id, log_remote=False):
        self.log_file = log_file_path
        self.log_remote = log_remote
        self.battery_monitor_available = True

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
        self.location = LocationInfo('Melbourne', 'Australia', 'Australia/Melbourne',
                                     latitude=-37.8136, longitude=144.9631)
        self.timezone = pytz.timezone(self.location.timezone)
        self.ensure_log_file_exists()

        if self.log_remote:
            self.db = firestore.Client(project=project_id)
            self.storage_client = storage.Client(project=project_id)

    def log_battery_to_firestore(self, battery_voltage, status):
        """Log detection results to Firestore."""
        if not self.log_remote:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        doc_ref = self.db.collection('battery').document(timestamp)

        doc_data = {
            "battery_voltage": battery_voltage,
            "status": status
        }

        doc_ref.set(doc_data)

    def ensure_log_file_exists(self):
        """Create the log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
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

        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if voltage is not None:
                    writer.writerow([timestamp, f"{voltage:.2f}", shutdown_reason or "on"])
                    print(f"Logged: Time: {timestamp}, Voltage: {voltage:.2f}V")
                else:
                    writer.writerow([timestamp, "N/A", shutdown_reason or "on (no battery data)"])
                    print(f"Logged: Time: {timestamp}, Voltage: N/A (battery monitor unavailable)")

            if self.log_remote and voltage is not None:
                self.log_battery_to_firestore(voltage, shutdown_reason or "on")
            elif self.log_remote:
                self.log_battery_to_firestore(None, shutdown_reason or "on (no battery data)")
        except Exception as e:
            print(f"Error logging data: {e}")

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

        # Try shutdown commands in order of preference
        shutdown_commands = [
            ['sudo', 'shutdown', '-h', 'now'],
            ['sudo', 'poweroff'],
            ['sudo', 'halt', '-p']
        ]

        for cmd in shutdown_commands:
            try:
                subprocess.run(cmd, check=True)
                time.sleep(2)  # Give the command time to take effect
            except subprocess.CalledProcessError as e:
                print(f"Failed to shutdown with {cmd}: {e}")
                continue

        # If we get here, none of the shutdown commands worked
        print("All shutdown attempts failed!")
        sys.exit(1)

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
        if current_voltage < 3.20:
            self.perform_shutdown("Shutdown! Low battery!")


def main():
    args = parse_arguments()

    try:
        monitor = BatteryMonitor(
            log_file_path=args.log_file_path,
            project_id=args.project_id,
            log_remote=args.log_remote,
        )
        time.sleep(5)

        print(f"Battery monitoring started. Logging to {monitor.log_file}")
        if not monitor.battery_monitor_available:
            print("Running in fallback mode without battery monitoring")

        while True:
            monitor.log_data()
            monitor.check_shutdown_condition()

            # Wait for specified minutes
            time.sleep(args.log_rate_min * 60)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()