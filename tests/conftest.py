"""Tests that pin data living only in the archive repository are skipped here.

The archive (github.com/EmilySmithCreate/RecreationalPhysics) holds results/, the public pages, the paper drafts
and the cloud queue; this working copy does not (README). The tests below read those files and are the archive's
to run. They are skipped by name so that the test files stay identical to the archive's and can be synced.
"""

import pytest

ARCHIVE_ONLY_FILES = {
    "test_main_nerd_pages.py",      # the public blog pages (docs/public/mainnerd)
    "test_read_column_wiring.py",   # saved wirings in results/
    "test_read_leftover.py",        # saved end states in results/
    "test_read_shapes.py",          # saved end states in results/
    "test_t9_squares.py",           # every T9 run in results/
    "test_t58_sweeps.py",           # the saved T58 traces in results/
    "test_curled_revision.py",      # a check script beside paper 1
}

ARCHIVE_ONLY_TESTS = {
    "test_across_saved_wirings_the_action_acts_like_the_energy_below_lambda_one",   # saved wirings
    "test_the_queue_file_names_every_config_once",                                    # cloud/queue
    "test_the_thirty_configs_and_the_queue_file",                                     # cloud/queue
    "test_the_held_out_configs_change_only_name_sizes_seed_purpose_and_written",     # cloud/queue
}

REASON = "reads results, pages or the cloud queue that live only in the archive repository"


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in ARCHIVE_ONLY_FILES or item.originalname in ARCHIVE_ONLY_TESTS:
            item.add_marker(pytest.mark.skip(reason=REASON))
