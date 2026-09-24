"""The rytp schema, as an ordered list of migrations (contracts §3).

Three rules, none of them negotiable:

* **One migration per table.** Small diffs review better than one
  500-line CREATE block, and a later version can add a column without
  disturbing what shipped.
* **Never edit a tuple that has run.** A migration that has executed on
  the owner's machine is history. Append a new one instead.
* **`schema_version` is not a migration.** The runner needs somewhere to
  record its progress before it can run anything, so it bootstraps that
  table itself.

The DDL text is copied verbatim from contracts §3. The only liberty
taken is ordering: `speakers` is created before `video_speakers` so the
foreign key is not a forward reference.
"""

from __future__ import annotations

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE channels (
            id              INTEGER PRIMARY KEY,
            url             TEXT NOT NULL UNIQUE,
            title           TEXT NOT NULL,
            last_synced_at  TEXT
        );
        """,
    ),
    (
        2,
        """
        CREATE TABLE videos (
            id            INTEGER PRIMARY KEY,
            source        TEXT NOT NULL CHECK (source IN ('youtube','ytdlp','local')),
            kind          TEXT NOT NULL CHECK (kind IN ('video','short','livestream','other')),
            channel_id    INTEGER REFERENCES channels(id),
            external_id   TEXT,
            url           TEXT,
            title         TEXT NOT NULL,
            duration_ms   INTEGER,
            published_at  TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL,
            UNIQUE (source, external_id)
        );
        CREATE INDEX videos_channel ON videos(channel_id);
        CREATE INDEX videos_source  ON videos(source);
        """,
    ),
    (
        3,
        """
        CREATE TABLE assets (
            id          INTEGER PRIMARY KEY,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK (role IN ('audio','video','captions','container')),
            format_id   TEXT,
            path        TEXT NOT NULL,
            bytes       INTEGER,
            width       INTEGER,
            height      INTEGER,
            abr         REAL,
            acquired_at TEXT NOT NULL
        );
        CREATE INDEX assets_video_role ON assets(video_id, role);
        -- A video has at most one audio asset and at most one captions asset.
        -- Video renditions are unconstrained: many per video is the point.
        CREATE UNIQUE INDEX assets_one_audio    ON assets(video_id) WHERE role = 'audio';
        CREATE UNIQUE INDEX assets_one_captions ON assets(video_id) WHERE role = 'captions';
        """,
    ),
    (
        4,
        """
        CREATE TABLE speakers (
            id           INTEGER PRIMARY KEY,
            label        TEXT NOT NULL UNIQUE,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            notes        TEXT,
            created_at   TEXT NOT NULL
        );
        """,
    ),
    (
        5,
        """
        CREATE TABLE video_speakers (
            id          INTEGER PRIMARY KEY,
            video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            local_label TEXT NOT NULL,
            speaker_id  INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
            embedding   BLOB,
            engine      TEXT NOT NULL,   -- which diarizer produced this label
            UNIQUE (video_id, local_label)
        );
        CREATE INDEX video_speakers_speaker ON video_speakers(speaker_id);
        """,
    ),
    (
        6,
        """
        CREATE TABLE words (
            id               INTEGER PRIMARY KEY,
            video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            ord              INTEGER NOT NULL,
            start_ms         INTEGER NOT NULL,
            end_ms           INTEGER,
            text             TEXT NOT NULL,
            normalized_text  TEXT NOT NULL,
            stem             TEXT NOT NULL,
            confidence       REAL,
            align_score      REAL,
            source           TEXT NOT NULL CHECK (source IN ('caption','timed','aligned')),
            engine           TEXT NOT NULL,
            video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
            UNIQUE (video_id, ord),
            CHECK (source = 'caption' OR end_ms IS NOT NULL)
        );
        CREATE INDEX words_video_ord  ON words(video_id, ord);
        CREATE INDEX words_normalized ON words(normalized_text);
        -- The assembler only ever looks at cuttable words, and most of the corpus
        -- will be caption-tier. Without this the hot lookup reads every row matching
        -- a token and filters afterwards, so its cost scales with the whole corpus
        -- rather than with the alignable part of it.
        CREATE INDEX words_alignable ON words(normalized_text) WHERE source = 'aligned';
        CREATE INDEX words_stem       ON words(stem);
        CREATE INDEX words_speaker    ON words(video_speaker_id);
        """,
    ),
    (
        7,
        """
        CREATE TABLE utterances (
            id               INTEGER PRIMARY KEY,
            video_id         INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            video_speaker_id INTEGER REFERENCES video_speakers(id) ON DELETE SET NULL,
            start_ms         INTEGER NOT NULL,
            end_ms           INTEGER NOT NULL,
            first_word_ord   INTEGER NOT NULL,
            last_word_ord    INTEGER NOT NULL,
            text             TEXT NOT NULL,
            normalized_text  TEXT NOT NULL,
            stem_text        TEXT NOT NULL
        );
        CREATE INDEX utterances_video ON utterances(video_id, start_ms);
        """,
    ),
    (
        8,
        """
        CREATE VIRTUAL TABLE utterances_fts USING fts5(
            normalized_text, stem_text,
            content='utterances', content_rowid='id',
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TRIGGER utterances_ai AFTER INSERT ON utterances BEGIN
            INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
            VALUES (new.id, new.normalized_text, new.stem_text);
        END;
        CREATE TRIGGER utterances_ad AFTER DELETE ON utterances BEGIN
            INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
            VALUES ('delete', old.id, old.normalized_text, old.stem_text);
        END;
        CREATE TRIGGER utterances_au AFTER UPDATE ON utterances BEGIN
            INSERT INTO utterances_fts(utterances_fts, rowid, normalized_text, stem_text)
            VALUES ('delete', old.id, old.normalized_text, old.stem_text);
            INSERT INTO utterances_fts(rowid, normalized_text, stem_text)
            VALUES (new.id, new.normalized_text, new.stem_text);
        END;
        """,
    ),
    (
        9,
        """
        CREATE TABLE video_acoustics (
            video_id             INTEGER PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
            f0_mean              REAL,
            f0_std               REAL,
            spectral_tilt        REAL,
            noise_floor_db       REAL,
            reverb_proxy         REAL,
            loudness_lufs        REAL,
            computed_at          TEXT NOT NULL
        );
        """,
    ),
    (
        10,
        """
        CREATE TABLE jobs (
            id           INTEGER PRIMARY KEY,
            kind         TEXT NOT NULL,
            target_id    INTEGER NOT NULL,
            state        TEXT NOT NULL CHECK (state IN
                           ('pending','running','done','failed','blocked','cancelled')),
            pool         TEXT NOT NULL CHECK (pool IN ('network','gpu','cpu')),
            priority     INTEGER NOT NULL DEFAULT 0,
            attempts     INTEGER NOT NULL DEFAULT 0,
            not_before   TEXT,
            last_error   TEXT,
            note         TEXT,          -- non-fatal finding returned by the handler
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at   TEXT NOT NULL,
            started_at   TEXT,
            finished_at  TEXT,
            UNIQUE (kind, target_id)
        );
        CREATE INDEX jobs_claim ON jobs(state, pool, priority, not_before);
        """,
    ),
    (
        11,
        """
        CREATE TABLE settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        -- `default_aligner` names the aligner that `ingest --transcribe` stamps onto
        -- the `align` jobs it creates. Empty means no alignment: ingest enqueues no
        -- `align` job at all and the words stay `timed` until you align them by hand.
        -- A job enqueued with an unregistered aligner name is rejected at enqueue
        -- time, not after five failed retries.
        INSERT INTO settings (key, value) VALUES ('default_aligner', '');
        """,
    ),
    (
        12,
        """
        -- A render targets a cut list, which is a named file with no integer
        -- identity, and jobs.target_id is an INTEGER. This table gives a render
        -- that identity, and gives render history somewhere to live.
        CREATE TABLE renders (
            id           INTEGER PRIMARY KEY,
            cutlist_name TEXT NOT NULL,
            output_path  TEXT,
            canvas_mode  TEXT NOT NULL,
            state        TEXT NOT NULL CHECK (state IN ('planned','rendered','failed')),
            created_at   TEXT NOT NULL,
            finished_at  TEXT
        );
        CREATE INDEX renders_cutlist ON renders(cutlist_name);
        """,
    ),
]

#: The version a fully migrated database reports.
LATEST_VERSION: int = MIGRATIONS[-1][0]
