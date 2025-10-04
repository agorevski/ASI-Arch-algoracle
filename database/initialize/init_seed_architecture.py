#!/usr/bin/env python3
"""
Base class for initializing ASI-Arch with seed architectures
This provides shared functionality for adding baseline architectures to the database.
"""

import json
import logging
import os
import requests
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))
from config import Config

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class SeedArchitectureInitializer(ABC):
    """Base class for seed architecture initialization"""

    def __init__(self):
        self.api_base_url = Config.DATABASE

    @abstractmethod
    def get_train_result(self) -> str:
        """Return the training result data"""
        pass

    @abstractmethod
    def get_test_result(self) -> str:
        """Return the test result data"""
        pass

    @abstractmethod
    def get_analysis(self) -> str:
        """Return the architecture analysis"""
        pass

    @abstractmethod
    def get_cognition(self) -> str:
        """Return the research context and cognition"""
        pass

    @abstractmethod
    def get_log(self) -> str:
        """Return the training log"""
        pass

    @abstractmethod
    def get_motivation(self) -> str:
        """Return the research motivation"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the seed element name"""
        pass

    @abstractmethod
    def get_summary(self) -> str:
        """Return the seed element summary"""
        pass

    @abstractmethod
    def get_source_path(self) -> str:
        """Return the seed element source path"""
        pass

    def get_pipeline_path(self) -> Path:
        """Return the path to the pipeline directory. Override if different."""
        return Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))

    def get_display_name(self) -> str:
        """Return the display name for console output. Override if different."""
        return "seed architecture"

    def read_source_file(self) -> str:
        """Read the source code"""
        source_path = self.get_source_path()
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with open(source_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_seed_element(self) -> dict:
        """Create the seed data element"""
        current_time = datetime.now().isoformat()

        program = self.read_source_file()

        result = {
            "train": self.get_train_result(),
            "test": self.get_test_result()
        }

        return {
            "time": current_time,
            "name": self.get_name(),
            "result": result,
            "program": program,
            "analysis": self.get_analysis(),
            "cognition": self.get_cognition(),
            "log": self.get_log(),
            "motivation": self.get_motivation(),
            "summary": self.get_summary()
        }

    async def add_seed_to_database(self) -> bool:
        """Add the seed element to the database via API"""

        logging.info(f"Creating seed element: {self.get_name()}...")
        element = self.create_seed_element()

        # API endpoint
        url = f"{self.api_base_url}/elements"

        try:
            print("Sending seed element to database...")
            response = requests.post(url, json=element, timeout=30)

            if response.status_code == 200:
                result = response.json()
                print("✅ Seed element added successfully!")
                print(f"   Element ID: {result.get('message', 'Added')}")
                return True
            else:
                print(f"❌ Failed to add seed element: {response.status_code}")
                print(f"   Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to database API: {e}")
            return False

    def update_candidate_storage(self) -> bool:
        """Update the candidate storage JSON file"""
        candidate_file = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'candidate_storage.json'))

        try:
            # Read current candidate storage
            with open(candidate_file, 'r') as f:
                storage = json.load(f)

            # Update with seed information
            storage["candidates"] = [1]  # Index 1 will be the seed element
            storage["new_data_count"] = 1
            storage["last_updated"] = datetime.now().isoformat()

            # Write back
            with open(candidate_file, 'w') as f:
                json.dump(storage, f, indent=2)

            print("✅ Updated candidate_storage.json")
            return True

        except Exception as e:
            print(f"❌ Failed to update candidate storage: {e}")
            return False

    async def run(self) -> bool:
        """Main initialization function"""

        print(f"🚀 Initializing ASI-Arch with seed {self.get_display_name()}")
        print("=" * 60)

        # Check if database API is running
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"📊 Database Status: {stats['total_records']} records")
            else:
                print("❌ Database API not responding properly")
                return False
        except requests.exceptions.RequestException:
            print("❌ Database API not accessible. Please start the database service first.")
            print("   Run: cd database && ./start_api.sh")
            return False

        # Add seed element to database
        success = await self.add_seed_to_database()
        if not success:
            return False

        # Update candidate storage
        success = self.update_candidate_storage()
        if not success:
            return False

        # Verify the addition
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                print(f"📊 Updated Database: {stats['total_records']} records")

            response = requests.get(f"{self.api_base_url}/candidates/all", timeout=5)
            if response.status_code == 200:
                candidates = response.json()
                print(f"🎯 Candidate Pool: {len(candidates)} candidates")
        except Exception as e:
            print(f"❌ Error verifying addition: {e}")
            return False

        print("=" * 60)
        print("✅ ASI-Arch initialization complete!")
        print("   You can now run experiments with: cd pipeline && python pipeline.py")

        return True
