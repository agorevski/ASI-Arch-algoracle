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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


class SeedArchitectureInitializer(ABC):
    """Base class for seed architecture initialization"""

    def __init__(self):
        """Initialize the seed architecture initializer.

        Sets up the API base URL from the configuration for database communication.
        """
        self.api_base_url = Config.DATABASE

    @abstractmethod
    def get_train_result(self) -> str:
        """Return the training result data.

        Returns:
            str: The training result data as a string representation.
        """
        pass

    @abstractmethod
    def get_test_result(self) -> str:
        """Return the test result data.

        Returns:
            str: The test result data as a string representation.
        """
        pass

    @abstractmethod
    def get_analysis(self) -> str:
        """Return the architecture analysis.

        Returns:
            str: The analysis of the architecture's design and capabilities.
        """
        pass

    @abstractmethod
    def get_cognition(self) -> str:
        """Return the research context and cognition.

        Returns:
            str: The research context and cognitive aspects of the architecture.
        """
        pass

    @abstractmethod
    def get_log(self) -> str:
        """Return the training log.

        Returns:
            str: The training log containing execution details and metrics.
        """
        pass

    @abstractmethod
    def get_motivation(self) -> str:
        """Return the research motivation.

        Returns:
            str: The motivation and rationale behind the architecture research.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the seed element name.

        Returns:
            str: The unique name identifier for the seed element.
        """
        pass

    @abstractmethod
    def get_summary(self) -> str:
        """Return the seed element summary.

        Returns:
            str: A brief summary describing the seed element.
        """
        pass

    @abstractmethod
    def get_source_path(self) -> str:
        """Return the seed element source path.

        Returns:
            str: The file system path to the source code file.
        """
        pass

    def get_pipeline_path(self) -> Path:
        """Return the path to the pipeline directory.

        Override this method if the pipeline directory is in a different location.

        Returns:
            Path: The path to the pipeline directory.
        """
        return Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))

    def get_display_name(self) -> str:
        """Return the display name for console output.

        Override this method to provide a custom display name for logging.

        Returns:
            str: The human-readable display name for console output.
        """
        return "seed architecture"

    def read_source_file(self) -> str:
        """Read the source code from the file system.

        Returns:
            str: The contents of the source code file.

        Raises:
            FileNotFoundError: If the source file does not exist at the specified path.
        """
        source_path = self.get_source_path()
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        with open(source_path, 'r', encoding='utf-8') as f:
            return f.read()

    def create_seed_element(self) -> dict:
        """Create the seed data element.

        Assembles all seed element components into a dictionary structure
        suitable for database storage.

        Returns:
            dict: A dictionary containing the complete seed element with keys:
                time, name, result, program, analysis, cognition, log,
                motivation, and summary.
        """
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
        """Add the seed element to the database via API.

        Creates the seed element and sends it to the database API endpoint.

        Returns:
            bool: True if the seed element was added successfully, False otherwise.
        """
        logging.info(f"Creating seed element: {self.get_name()}...")
        element = self.create_seed_element()

        # API endpoint
        url = f"{self.api_base_url}/elements"

        try:
            logging.info("Sending seed element to database...")
            response = requests.post(url, json=element, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logging.info("✅ Seed element added successfully!")
                logging.info(f"   Element ID: {result.get('message', 'Added')}")
                return True
            else:
                logging.error(f"❌ Failed to add seed element: {response.status_code}")
                logging.error(f"   Response: {response.text}")
                return False

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error connecting to database API: {e}")
            return False

    def update_candidate_storage(self) -> bool:
        """Update the candidate storage JSON file.

        Updates the candidate_storage.json file with the new seed element
        information, including candidate indices and timestamp.

        Returns:
            bool: True if the candidate storage was updated successfully,
                False otherwise.
        """
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

            logging.info("✅ Updated candidate_storage.json")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to update candidate storage: {e}")
            return False

    async def run(self) -> bool:
        """Main initialization function.

        Executes the complete initialization workflow including database
        connectivity check, seed element addition, and candidate storage update.

        Returns:
            bool: True if initialization completed successfully, False otherwise.
        """
        logging.info(f"🚀 Initializing ASI-Arch with seed {self.get_display_name()}")
        logging.info("=" * 60)

        # Check if database API is running
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                logging.info(f"📊 Database Status: {stats['total_records']} records")
            else:
                logging.error("❌ Database API not responding properly")
                return False
        except requests.exceptions.RequestException:
            logging.error("❌ Database API not accessible. Please start the database service first.")
            logging.error("   Run: cd database && ./start_api.sh")
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
                logging.info(f"📊 Updated Database: {stats['total_records']} records")

            response = requests.get(f"{self.api_base_url}/candidates/all", timeout=5)
            if response.status_code == 200:
                candidates = response.json()
                logging.info(f"🎯 Candidate Pool: {len(candidates)} candidates")
        except Exception as e:
            logging.error(f"❌ Error verifying addition: {e}")
            return False

        logging.info("=" * 60)
        logging.info("✅ ASI-Arch initialization complete!")
        logging.info("   You can now run experiments with: cd pipeline && python pipeline.py")

        return True

