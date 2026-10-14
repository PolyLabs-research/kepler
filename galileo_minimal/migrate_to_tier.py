#!/usr/bin/env python3
"""
migrate_to_tier.py - Automated migration helper for existing checkpoints and data

This script helps migrate existing checkpoints and data to the tier structure.

Usage:
    python migrate_to_tier.py --checkpoints  # Migrate checkpoints to tier1/
    python migrate_to_tier.py --data         # Migrate data to tier1/
    python migrate_to_tier.py --all          # Migrate both
    python migrate_to_tier.py --dry-run      # Show what would be done
"""

import argparse
import shutil
import json
from pathlib import Path

def migrate_checkpoints(dry_run=False):
    """Migrate existing checkpoints to tier1/ subdirectories."""
    print("=" * 70)
    print("Migrating Checkpoints to Tier Structure")
    print("=" * 70)
    
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        print("No checkpoints/ directory found. Nothing to migrate.")
        return
    
    migrated = 0
    
    for domain_dir in ckpt_dir.iterdir():
        if not domain_dir.is_dir():
            continue
            
        for model_dir in domain_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Check if already has tier structure
            tier_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("tier")]
            if tier_dirs:
                print(f"✓ {model_dir.relative_to(ckpt_dir)} already has tier structure")
                continue
            
            # Check for checkpoint files at root
            checkpoint_files = ["model_best.pt", "model_final.pt", "model_interrupted.pt", "vocab.json"]
            found_files = [f for f in checkpoint_files if (model_dir / f).exists()]
            
            if not found_files:
                continue
            
            # Create tier1 directory
            tier1_dir = model_dir / "tier1"
            
            print(f"\n📦 {model_dir.relative_to(ckpt_dir)}")
            print(f"   Found files: {', '.join(found_files)}")
            
            if dry_run:
                print(f"   Would create: {tier1_dir.relative_to(ckpt_dir)}")
                for f in found_files:
                    print(f"   Would move: {f} -> tier1/{f}")
            else:
                tier1_dir.mkdir(exist_ok=True)
                print(f"   Created: {tier1_dir.relative_to(ckpt_dir)}")
                
                for f in found_files:
                    src = model_dir / f
                    dst = tier1_dir / f
                    shutil.move(str(src), str(dst))
                    print(f"   Moved: {f} -> tier1/{f}")
                
                migrated += 1
    
    print("\n" + "=" * 70)
    if dry_run:
        print(f"Dry run complete. Would migrate {migrated} model(s).")
    else:
        print(f"Migration complete. Migrated {migrated} model(s).")
    print("=" * 70)

def migrate_data(dry_run=False):
    """Migrate existing binary data to tier1/ subdirectories."""
    print("=" * 70)
    print("Migrating Data to Tier Structure")
    print("=" * 70)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("No data/ directory found. Nothing to migrate.")
        return
    
    migrated = 0
    
    for model_dir in data_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Check if already has tier structure
        tier_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("tier")]
        if tier_dirs:
            print(f"✓ {model_dir.relative_to(data_dir)} already has tier structure")
            continue
        
        # Check for data files at root
        data_files = ["features.bin", "targets.bin", "metadata.json"]
        found_files = [f for f in data_files if (model_dir / f).exists()]
        
        if not found_files:
            continue
        
        # Create tier1 directory
        tier1_dir = model_dir / "tier1"
        
        print(f"\n📦 {model_dir.relative_to(data_dir)}")
        print(f"   Found files: {', '.join(found_files)}")
        
        if dry_run:
            print(f"   Would create: {tier1_dir.relative_to(data_dir)}")
            for f in found_files:
                print(f"   Would move: {f} -> tier1/{f}")
        else:
            tier1_dir.mkdir(exist_ok=True)
            print(f"   Created: {tier1_dir.relative_to(data_dir)}")
            
            for f in found_files:
                src = model_dir / f
                dst = tier1_dir / f
                shutil.move(str(src), str(dst))
                print(f"   Moved: {f} -> tier1/{f}")
            
            # Update metadata.json to include tier field
            metadata_path = tier1_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                if "tier" not in metadata:
                    metadata["tier"] = 1
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"   Updated metadata.json with tier=1")
            
            migrated += 1
    
    print("\n" + "=" * 70)
    if dry_run:
        print(f"Dry run complete. Would migrate {migrated} model(s).")
    else:
        print(f"Migration complete. Migrated {migrated} model(s).")
    print("=" * 70)

def create_backup():
    """Create a timestamped backup of checkpoints and data."""
    from datetime import datetime
    
    backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir = Path(backup_name)
    backup_dir.mkdir(exist_ok=True)
    
    print(f"Creating backup: {backup_name}/")
    
    backed_up = []
    
    if Path("checkpoints").exists():
        shutil.copytree("checkpoints", backup_dir / "checkpoints")
        backed_up.append("checkpoints")
        print(f"  ✓ Backed up checkpoints/")
    
    if Path("data").exists():
        shutil.copytree("data", backup_dir / "data")
        backed_up.append("data")
        print(f"  ✓ Backed up data/")
    
    if backed_up:
        print(f"\n✓ Backup complete: {backup_name}/")
        print(f"  Contains: {', '.join(backed_up)}")
        return backup_name
    else:
        backup_dir.rmdir()
        print("Nothing to back up (no checkpoints/ or data/ directories found)")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing checkpoints and data to tier structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_to_tier.py --dry-run --all      # Preview migration
  python migrate_to_tier.py --backup --all       # Create backup then migrate
  python migrate_to_tier.py --checkpoints         # Migrate only checkpoints
  python migrate_to_tier.py --data                # Migrate only data
        """
    )
    
    parser.add_argument("--checkpoints", action="store_true", help="Migrate checkpoints to tier1/")
    parser.add_argument("--data", action="store_true", help="Migrate data to tier1/")
    parser.add_argument("--all", action="store_true", help="Migrate both checkpoints and data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true", help="Create timestamped backup before migrating")
    
    args = parser.parse_args()
    
    # Default to --all if nothing specified
    if not (args.checkpoints or args.data or args.all):
        args.all = True
    
    if args.all:
        args.checkpoints = True
        args.data = True
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_name = create_backup()
        if backup_name:
            print(f"\n💾 Backup created. You can restore with:")
            print(f"   rm -rf checkpoints/ data/")
            print(f"   cp -r {backup_name}/* .")
        print()
    
    # Perform migrations
    if args.checkpoints:
        migrate_checkpoints(args.dry_run)
        print()
    
    if args.data:
        migrate_data(args.dry_run)
        print()
    
    # Final message
    if args.dry_run:
        print("Dry run complete. Run without --dry-run to perform migration.")
    else:
        print("✓ Migration complete!")
        print("\nNext steps:")
        print("  1. Verify: python validate_migration.py")
        print("  2. Test: python test.py --model polynomial --tier 1")
        print("  3. Train new tiers: python train.py --model polynomial --tier 2 --steps 5000")

if __name__ == "__main__":
    main()
