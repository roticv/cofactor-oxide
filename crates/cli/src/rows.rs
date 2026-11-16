//! GNFS relations collected in memory, or with their rows in a file
//! beside `--save-relations FILE`, `FILE.rows`, so that memory holds only
//! the pairs seen and the columns of the large ideals while sieving, and
//! only the rows that survive pruning while filtering.

use std::fs::File;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use cofox_core::nfs::{DiskRelations, Relation, Relations, Rows};

/// The relations of a GNFS run, wherever their rows are.
pub enum Collection {
    Memory(Relations<16>),
    Disk(DiskRelations<16>, RowFile),
}

impl Collection {
    /// Adds `relations`, skipping any already found.
    pub fn add(&mut self, relations: impl IntoIterator<Item = Relation>) {
        match self {
            Self::Memory(r) => r.add(relations),
            Self::Disk(r, rows) => r.add(relations, rows),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Memory(r) => r.len(),
            Self::Disk(r, _) => r.len(),
        }
    }

    /// Whether there are enough relations, checked now and then.
    pub fn is_ready(&mut self) -> bool {
        match self {
            Self::Memory(r) => r.is_ready(),
            Self::Disk(r, rows) => r.is_ready(rows),
        }
    }
}

/// The rows of [`DiskRelations`] in a file, each its word count and then
/// its words, little-endian. The file is emptied when opened and removed
/// when dropped: a resumed run writes it again from the relations it
/// loads.
pub struct RowFile {
    path: PathBuf,
    writer: BufWriter<File>,
    bytes: Vec<u8>,
}

impl RowFile {
    /// `FILE.rows` for `--save-relations FILE`, emptied.
    pub fn create(save: &Path) -> std::io::Result<Self> {
        let mut path = save.as_os_str().to_owned();
        path.push(".rows");
        let path = PathBuf::from(path);
        let file = File::create(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::with_capacity(1 << 20, file),
            bytes: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn fail(&self, doing: &str, e: &std::io::Error) -> ! {
        eprintln!("cofox: can't {doing} {}: {e}", self.path.display());
        std::process::exit(1);
    }
}

impl Rows for RowFile {
    fn append(&mut self, words: &[u32]) {
        self.bytes.clear();
        self.bytes
            .extend_from_slice(&(words.len() as u32).to_le_bytes());
        self.bytes
            .extend(words.iter().flat_map(|w| w.to_le_bytes()));
        if let Err(e) = self.writer.write_all(&self.bytes) {
            self.fail("write the matrix's rows to", &e);
        }
    }

    fn read(&mut self, f: &mut dyn FnMut(&[u32])) {
        if let Err(e) = self.writer.flush() {
            self.fail("write the matrix's rows to", &e);
        }
        let file = File::open(&self.path).unwrap_or_else(|e| self.fail("read back", &e));
        let mut reader = BufReader::with_capacity(8 << 20, file);
        let (mut bytes, mut words) = (Vec::new(), Vec::new());
        let mut len = [0u8; 4];
        loop {
            match reader.read_exact(&mut len) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => self.fail("read back", &e),
            }
            bytes.resize(4 * u32::from_le_bytes(len) as usize, 0);
            if let Err(e) = reader.read_exact(&mut bytes) {
                self.fail("read back", &e);
            }
            words.clear();
            words.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| u32::from_le_bytes(b)),
            );
            f(&words);
        }
    }
}

impl Drop for RowFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
