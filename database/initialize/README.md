# Database Initialize

This directory contains utilities for initializing and managing the ASI-Arch database with seed architectures.

## Overview

The ASI-Arch system requires seed architectures to bootstrap the evolutionary process. These scripts provide functionality to:

- Clear all database entries (for fresh starts or resets)
- Seed the database with baseline architectures (DeltaNet, InvisMark)
- Establish initial candidate pools for evolution

## Files

### `delete_all_database_entries.py`

**Purpose**: Complete database cleanup utility that removes all entries from MongoDB, FAISS index, and candidate storage.

**Features**:

- Database connection validation
- Safe deletion with confirmation prompt
- Clears MongoDB records, FAISS embeddings, and candidate storage
- Post-deletion verification

**Usage**:

```bash
cd database/initialize
python delete_all_database_entries.py
```

**Safety**: Requires typing `DELETE` to confirm the operation. This action is irreversible.

---

## Seed Implementation

### `init_seed_architecture.py` - Abstract base class providing shared functionality for seeding architectures

### `seed_deltanet.py` - Initialize the DeltaNet linear attention architecture

### `seed_invismark.py` - Initialize the InvisMark image watermarking architecture

**Usage**:

```bash
cd database/initialize
python seed_deltanet.py
python seed_invismark.py
```

## Generate a new seed quickly

Here is a helpful prompt written for Cline to help seed a new approach

```prompt
@/init_seed_architecture.py @/seed_invismark.py
Please create a new file called <filename> that extends SeedArchitectureInitializer similar to InvisMarkSeeder but instead please reference
 @/your_arxiv_paper_here.pdf aka @/cognition_base\cognition\[model]\your_arxiv_paper_here.json instead
```

Submit to Claude Sonnet 4.5 (at time of writing) as a Plan, generate a plan, then Act on the plan

## Safety Notes

⚠️ **Warning**: The `delete_all_database_entries.py` script permanently removes all data. Always backup important experiments before running.

✅ **Best Practice**: Use version control and save architecture results before resetting the database.

🔄 **Idempotent Seeding**: Running seed scripts multiple times will create duplicate entries. Clear the database first if you need a fresh start.

---

## Related Documentation

- **Database API**: `database/README.md` (if exists) or `database/mongodb_api.py`
- **Pipeline**: `pipeline/README.md` for running experiments
- **Config**: `pipeline/config.py` for database connection settings
- **Architecture Pool**: `pipeline/pool/` for architecture implementations

---

## License

Part of the ASI-Arch project. See root LICENSE file for details.
