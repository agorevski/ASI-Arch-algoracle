#!/usr/bin/env python3
"""
Delete all entries from the ASI-Arch database
This script clears all data from MongoDB, FAISS index, and candidate storage.
"""

import json
import logging
import os
import requests
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatabaseDeleter:
    """Utility class for deleting all database entries"""

    def __init__(self):
        self.api_base_url = Config.DATABASE
        self.candidate_storage_file = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'candidate_storage.json'))

    def check_database_connection(self) -> bool:
        """Check if database API is accessible"""
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"📊 Database Status: {stats['total_records']} records found")
                return True
            else:
                print("❌ Database API not responding properly")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Database API not accessible: {e}")
            print("   Please start the database service first.")
            print("   Run: cd database && ./start_api.sh")
            return False

    def delete_all_elements(self) -> bool:
        """Delete all elements from the database via API"""
        url = f"{self.api_base_url}/elements/all"

        try:
            print("\n🗑️  Deleting all database entries...")
            response = requests.delete(url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print("✅ All database entries deleted successfully!")
                print(f"   Message: {result.get('message', 'Deleted')}")
                return True
            else:
                print(f"❌ Failed to delete entries: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to database API: {e}")
            return False

    def clear_candidate_storage(self) -> bool:
        """Clear the candidate storage JSON file"""
        try:
            # Reset candidate storage to empty state
            empty_storage = {
                "candidates": [],
                "new_data_count": 0,
                "last_updated": datetime.now().isoformat()
            }

            with open(self.candidate_storage_file, 'w') as f:
                json.dump(empty_storage, f, indent=2)

            print("✅ Candidate storage cleared")
            return True

        except Exception as e:
            print(f"❌ Failed to clear candidate storage: {e}")
            return False

    def verify_deletion(self) -> bool:
        """Verify that all data has been deleted"""
        try:
            # Check database stats
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"\n📊 Verification - Database Stats:")
                print(f"   Total records: {stats['total_records']}")
                print(f"   Unique names: {stats['unique_names']}")

                if stats['total_records'] == 0:
                    print("   ✅ Database is empty")
                else:
                    print(f"   ⚠️  Warning: {stats['total_records']} records still present")
                    return False

            # Check candidate pool
            response = requests.get(f"{self.api_base_url}/candidates/all", timeout=5)
            if response.status_code == 200:
                candidates = response.json()
                print(f"   Candidate pool: {len(candidates)} candidates")

                if len(candidates) == 0:
                    print("   ✅ Candidate pool is empty")
                else:
                    print(f"   ⚠️  Warning: {len(candidates)} candidates still present")
                    return False

            return True

        except Exception as e:
            print(f"❌ Error verifying deletion: {e}")
            return False

    def run(self) -> bool:
        """Main deletion function"""
        print("🚀 ASI-Arch Database Deletion Utility")
        print("=" * 60)
        print("⚠️  WARNING: This will DELETE ALL data from the database!")
        print("=" * 60)

        # Check database connection
        if not self.check_database_connection():
            return False

        # Confirm deletion
        print("\n⚠️  Are you sure you want to proceed? This action cannot be undone!")
        print("   Type 'DELETE' to confirm:")
        
        try:
            confirmation = input().strip()
            if confirmation != "DELETE":
                print("❌ Deletion cancelled")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n❌ Deletion cancelled")
            return False

        # Delete all elements
        if not self.delete_all_elements():
            return False

        # Clear candidate storage
        if not self.clear_candidate_storage():
            print("⚠️  Warning: Candidate storage may not be properly cleared")

        # Verify deletion
        if not self.verify_deletion():
            print("\n⚠️  Warning: Deletion may not be complete")
            return False

        print("\n" + "=" * 60)
        print("✅ Database deletion complete!")
        print("   All entries have been removed from:")
        print("   - MongoDB collection")
        print("   - FAISS index")
        print("   - Candidate storage")
        print("=" * 60)

        return True


def main():
    """Entry point for the script"""
    deleter = DatabaseDeleter()
    success = deleter.run()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
