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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatabaseDeleter:
    """Utility class for deleting all database entries"""

    def __init__(self):
        """Initialize the DatabaseDeleter with API URL and storage file path.

        Sets up the base URL for the database API from config and determines
        the path to the candidate storage JSON file.
        """
        self.api_base_url = Config.DATABASE
        self.candidate_storage_file = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'candidate_storage.json'))

    def check_database_connection(self) -> bool:
        """Check if the database API is accessible.

        Attempts to connect to the database API and retrieve stats to verify
        the connection is working properly.

        Returns:
            bool: True if the database API is accessible and responding,
                False otherwise.
        """
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
        """Delete all elements from the database via API.

        Sends a DELETE request to the database API to remove all elements
        from the database.

        Returns:
            bool: True if all elements were successfully deleted,
                False if the deletion failed.
        """
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
        """Clear the candidate storage JSON file.

        Resets the candidate storage file to an empty state with zero
        candidates and updates the last_updated timestamp.

        Returns:
            bool: True if the candidate storage was successfully cleared,
                False if an error occurred.
        """
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
        """Verify that all data has been deleted.

        Checks the database stats and candidate pool to confirm that
        all data has been successfully removed.

        Returns:
            bool: True if the database and candidate pool are both empty,
                False if any data remains or verification fails.
        """
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
        """Execute the main deletion workflow.

        Performs the complete deletion process including connection check,
        user confirmation, element deletion, candidate storage clearing,
        and verification of the deletion.

        Returns:
            bool: True if all deletion steps completed successfully,
                False if any step failed or user cancelled.
        """
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
    """Entry point for the script.

    Creates a DatabaseDeleter instance, runs the deletion workflow,
    and exits with appropriate status code.

    Returns:
        None: Exits with code 0 on success, 1 on failure.
    """
    deleter = DatabaseDeleter()
    success = deleter.run()

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
