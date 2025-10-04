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
                logging.info(f"📊 Database Status: {stats['total_records']} records found")
                return True
            else:
                logging.error("❌ Database API not responding properly")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Database API not accessible: {e}")
            logging.error("   Please start the database service first.")
            logging.error("   Run: cd database && ./start_api.sh")
            return False

    def delete_all_elements(self) -> bool:
        """Delete all elements from the database via API"""
        url = f"{self.api_base_url}/elements/all"

        try:
            logging.info("\n🗑️  Deleting all database entries...")
            response = requests.delete(url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logging.info("✅ All database entries deleted successfully!")
                logging.info(f"   Message: {result.get('message', 'Deleted')}")
                return True
            else:
                logging.error(f"❌ Failed to delete entries: {response.status_code}")
                logging.error(f"   Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error connecting to database API: {e}")
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

            logging.info("✅ Candidate storage cleared")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to clear candidate storage: {e}")
            return False

    def verify_deletion(self) -> bool:
        """Verify that all data has been deleted"""
        try:
            # Check database stats
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                logging.info(f"\n📊 Verification - Database Stats:")
                logging.info(f"   Total records: {stats['total_records']}")
                logging.info(f"   Unique names: {stats['unique_names']}")

                if stats['total_records'] == 0:
                    logging.info("   ✅ Database is empty")
                else:
                    logging.warning(f"   ⚠️  Warning: {stats['total_records']} records still present")
                    return False

            # Check candidate pool
            response = requests.get(f"{self.api_base_url}/candidates/all", timeout=5)
            if response.status_code == 200:
                candidates = response.json()
                logging.info(f"   Candidate pool: {len(candidates)} candidates")

                if len(candidates) == 0:
                    logging.info("   ✅ Candidate pool is empty")
                else:
                    logging.warning(f"   ⚠️  Warning: {len(candidates)} candidates still present")
                    return False

            return True

        except Exception as e:
            logging.error(f"❌ Error verifying deletion: {e}")
            return False

    def run(self) -> bool:
        """Main deletion function"""
        logging.info("🚀 ASI-Arch Database Deletion Utility")
        logging.info("=" * 60)
        logging.warning("⚠️  WARNING: This will DELETE ALL data from the database!")
        logging.info("=" * 60)

        # Check database connection
        if not self.check_database_connection():
            return False

        # Confirm deletion
        logging.warning("\n⚠️  Are you sure you want to proceed? This action cannot be undone!")
        logging.info("   Type 'DELETE' to confirm:")
        
        try:
            confirmation = input().strip()
            if confirmation != "DELETE":
                logging.info("❌ Deletion cancelled")
                return False
        except (KeyboardInterrupt, EOFError):
            logging.info("\n❌ Deletion cancelled")
            return False

        # Delete all elements
        if not self.delete_all_elements():
            return False

        # Clear candidate storage
        if not self.clear_candidate_storage():
            logging.warning("⚠️  Warning: Candidate storage may not be properly cleared")

        # Verify deletion
        if not self.verify_deletion():
            logging.warning("\n⚠️  Warning: Deletion may not be complete")
            return False

        logging.info("\n" + "=" * 60)
        logging.info("✅ Database deletion complete!")
        logging.info("   All entries have been removed from:")
        logging.info("   - MongoDB collection")
        logging.info("   - FAISS index")
        logging.info("   - Candidate storage")
        logging.info("=" * 60)

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
