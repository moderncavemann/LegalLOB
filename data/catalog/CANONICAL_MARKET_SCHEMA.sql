-- Canonical local-only relational schema for extracted market rows.
--
-- The GitHub release includes this DDL but deliberately contains no real market
-- rows. Provider-controlled data must stay in the local warehouse.

CREATE TABLE IF NOT EXISTS lob_snapshot_l2 (
    dataset_id TEXT NOT NULL,
    row_idx INTEGER NOT NULL,
    timestamp_ns INTEGER,
    day_id INTEGER,
    segment_id INTEGER,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    source_row_idx INTEGER,
    ask_price_01 REAL NOT NULL, ask_size_01 REAL NOT NULL,
    ask_price_02 REAL NOT NULL, ask_size_02 REAL NOT NULL,
    ask_price_03 REAL NOT NULL, ask_size_03 REAL NOT NULL,
    ask_price_04 REAL NOT NULL, ask_size_04 REAL NOT NULL,
    ask_price_05 REAL NOT NULL, ask_size_05 REAL NOT NULL,
    ask_price_06 REAL NOT NULL, ask_size_06 REAL NOT NULL,
    ask_price_07 REAL NOT NULL, ask_size_07 REAL NOT NULL,
    ask_price_08 REAL NOT NULL, ask_size_08 REAL NOT NULL,
    ask_price_09 REAL NOT NULL, ask_size_09 REAL NOT NULL,
    ask_price_10 REAL NOT NULL, ask_size_10 REAL NOT NULL,
    bid_price_01 REAL NOT NULL, bid_size_01 REAL NOT NULL,
    bid_price_02 REAL NOT NULL, bid_size_02 REAL NOT NULL,
    bid_price_03 REAL NOT NULL, bid_size_03 REAL NOT NULL,
    bid_price_04 REAL NOT NULL, bid_size_04 REAL NOT NULL,
    bid_price_05 REAL NOT NULL, bid_size_05 REAL NOT NULL,
    bid_price_06 REAL NOT NULL, bid_size_06 REAL NOT NULL,
    bid_price_07 REAL NOT NULL, bid_size_07 REAL NOT NULL,
    bid_price_08 REAL NOT NULL, bid_size_08 REAL NOT NULL,
    bid_price_09 REAL NOT NULL, bid_size_09 REAL NOT NULL,
    bid_price_10 REAL NOT NULL, bid_size_10 REAL NOT NULL,
    PRIMARY KEY (dataset_id, row_idx),
    CHECK (ask_price_01 > bid_price_01),
    CHECK (ask_price_01 <= ask_price_02 AND ask_price_02 <= ask_price_03),
    CHECK (ask_price_03 <= ask_price_04 AND ask_price_04 <= ask_price_05),
    CHECK (ask_price_05 <= ask_price_06 AND ask_price_06 <= ask_price_07),
    CHECK (ask_price_07 <= ask_price_08 AND ask_price_08 <= ask_price_09),
    CHECK (ask_price_09 <= ask_price_10),
    CHECK (bid_price_01 >= bid_price_02 AND bid_price_02 >= bid_price_03),
    CHECK (bid_price_03 >= bid_price_04 AND bid_price_04 >= bid_price_05),
    CHECK (bid_price_05 >= bid_price_06 AND bid_price_06 >= bid_price_07),
    CHECK (bid_price_07 >= bid_price_08 AND bid_price_08 >= bid_price_09),
    CHECK (bid_price_09 >= bid_price_10),
    CHECK (ask_size_01 >= 0 AND ask_size_02 >= 0 AND ask_size_03 >= 0),
    CHECK (ask_size_04 >= 0 AND ask_size_05 >= 0 AND ask_size_06 >= 0),
    CHECK (ask_size_07 >= 0 AND ask_size_08 >= 0 AND ask_size_09 >= 0),
    CHECK (ask_size_10 >= 0),
    CHECK (bid_size_01 >= 0 AND bid_size_02 >= 0 AND bid_size_03 >= 0),
    CHECK (bid_size_04 >= 0 AND bid_size_05 >= 0 AND bid_size_06 >= 0),
    CHECK (bid_size_07 >= 0 AND bid_size_08 >= 0 AND bid_size_09 >= 0),
    CHECK (bid_size_10 >= 0)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_dataset_time
    ON lob_snapshot_l2(dataset_id, timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_snapshot_dataset_split
    ON lob_snapshot_l2(dataset_id, split, row_idx);

CREATE TABLE IF NOT EXISTS lob_event_l3 (
    dataset_id TEXT NOT NULL,
    event_idx INTEGER NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    day_id INTEGER NOT NULL,
    event_type INTEGER NOT NULL,
    order_id INTEGER,
    price REAL,
    size REAL NOT NULL CHECK (size >= 0),
    direction INTEGER CHECK (direction IN (-1, 1)),
    passive_side TEXT CHECK (passive_side IN ('buy', 'sell') OR passive_side IS NULL),
    aggressor_side TEXT CHECK (aggressor_side IN ('buy', 'sell') OR aggressor_side IS NULL),
    state_idx_before INTEGER,
    state_idx_after INTEGER,
    is_submission INTEGER NOT NULL CHECK (is_submission IN (0, 1)),
    is_cancel INTEGER NOT NULL CHECK (is_cancel IN (0, 1)),
    is_delete INTEGER NOT NULL CHECK (is_delete IN (0, 1)),
    is_visible_execution INTEGER NOT NULL CHECK (is_visible_execution IN (0, 1)),
    is_hidden_execution INTEGER NOT NULL CHECK (is_hidden_execution IN (0, 1)),
    is_cross_trade INTEGER NOT NULL CHECK (is_cross_trade IN (0, 1)),
    is_halt INTEGER NOT NULL CHECK (is_halt IN (0, 1)),
    PRIMARY KEY (dataset_id, event_idx)
);

CREATE INDEX IF NOT EXISTS idx_event_dataset_time
    ON lob_event_l3(dataset_id, timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_event_dataset_type
    ON lob_event_l3(dataset_id, event_type, event_idx);

CREATE TABLE IF NOT EXISTS dataset_transform (
    dataset_id TEXT PRIMARY KEY,
    fit_split TEXT NOT NULL,
    fit_start_idx INTEGER,
    fit_end_idx INTEGER,
    price_transform TEXT NOT NULL,
    volume_transform TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    source_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS exact_replay_check (
    dataset_id TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    compared_rows INTEGER NOT NULL,
    matched_rows INTEGER NOT NULL,
    match_rate REAL NOT NULL CHECK (match_rate BETWEEN 0 AND 1),
    known_exception_rows INTEGER NOT NULL DEFAULT 0,
    report_path TEXT,
    PRIMARY KEY (dataset_id, trading_date)
);
