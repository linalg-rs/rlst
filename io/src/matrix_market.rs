//! Matrix market reader/writer
//!
//! The functions [read_array_mm] and [read_coordinate_mm] can read the Matrix Market array and coordinate
//! file formats. Currently, matrices with special symmetry properties are not supported. Input
//! files must use the `general` property. To write out matrices the functions [write_array_mm] and
//! [write_coordinate_mm] are provided.

use rlst_dense::array::DynamicArray;
use rlst_dense::traits::RawAccessMut;
use rlst_dense::types::RlstScalar;
use rlst_dense::types::{RlstError, RlstResult};
use rlst_sparse::sparse::csr_mat::CsrMatrix;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};
use std::path::Path;

/// Definition of Matrix format types
#[derive(PartialEq, Clone, Copy)]
pub enum MatrixFormat {
    /// Matrices in coordinate format.
    /// Entries are described as `(i, j, data)` triplets with row `i`,
    /// column `j` and corresponding `data`. This is mainly used for
    /// sparse matrices.
    Coordinate,
    /// Matrices in array format. All entries are provided in column-major
    /// format. This format is mainly used for dense matrices.
    Array,
}

/// Definition of the data type.
#[derive(PartialEq, Clone, Copy)]
pub enum DataType {
    /// Real entries.
    Real,
    /// Integer entries (currently not supported).
    Integer,
    /// Complex entries.
    Complex,
}

/// Identifier trait to associate a matrix market identifier with a Rust type.
pub trait MmIdentifier {
    /// Matrix market type
    const MMTYPE: &'static str;
}

impl MmIdentifier for f32 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for f64 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for rlst_dense::types::c32 {
    const MMTYPE: &'static str = "complex";
}

impl MmIdentifier for rlst_dense::types::c64 {
    const MMTYPE: &'static str = "complex";
}

/// Matrix market symmetry type.
#[derive(PartialEq, Clone, Copy)]
pub enum SymmetryType {
    /// General matrices without symmetry.
    General,
    /// Symmetric matrices (currently not supported).
    Symmetric,
    /// Skew-symmetric matrices (currently not supported).
    Skew,
    /// Hermitian matrices (currently not supported).
    Hermitian,
}

/// A simple container to store the information section of a matrix market file.
pub struct MatrixMarketInfo {
    format: MatrixFormat,
    data_type: DataType,
    symmetry: SymmetryType,
}

/// Export a matrix in coordinate format.
///
/// This function requires objects to implement the [rlst_dense::traits::AijIterator] and
/// [rlst_dense::traits::Shape] traits. Any object satisfying these traits can be written
/// out with this function.
pub fn write_coordinate_mm<
    T: RlstScalar + MmIdentifier,
    Mat: rlst_dense::traits::AijIterator<Item = T> + rlst_dense::traits::Shape<2>,
>(
    mat: &Mat,
    fname: &str,
) -> RlstResult<()> {
    let output = File::create(fname);

    let count = mat.iter_aij().count();

    if let Ok(mut output) = output {
        writeln!(
            output,
            "%%MatrixMarket matrix coordinate {} general",
            T::MMTYPE
        )
        .unwrap();
        writeln!(output, "%").unwrap();
        writeln!(output, "{} {} {}", mat.shape()[0], mat.shape()[1], count).unwrap();
        for (row, col, data) in mat.iter_aij() {
            writeln!(output, "{} {} {} ", 1 + row, 1 + col, data).unwrap();
        }

        Ok(())
    } else {
        Err(RlstError::IoError(format!("Could not open file {}", fname)))
    }
}

/// Export a matrix in array format.
///
/// This function requires objects to implement the [rlst_dense::traits::DefaultIterator] and
/// [rlst_dense::traits::Shape] traits. Any object satisfying these traits can be written
/// out with this function.
pub fn write_array_mm<
    T: RlstScalar + MmIdentifier,
    Mat: rlst_dense::traits::DefaultIterator<Item = T> + rlst_dense::traits::Shape<2>,
>(
    mat: &Mat,
    fname: &str,
) -> RlstResult<()> {
    let output = File::create(fname);

    if let Ok(mut output) = output {
        writeln!(output, "%%MatrixMarket matrix array {} general", T::MMTYPE).unwrap();
        writeln!(output, "%").unwrap();
        writeln!(output, "{} {}", mat.shape()[0], mat.shape()[1]).unwrap();
        for value in mat.iter() {
            writeln!(output, "{}", value).unwrap();
        }

        Ok(())
    } else {
        Err(RlstError::IoError(format!("Could not open file {}", fname)))
    }
}

/// Read an array in matrix market format.
///
/// The function returns a [DynamicArray] object representing the data in the file.
/// Currently only `general` matrices are supported without special symmetry.
pub fn read_array_mm<T: RlstScalar>(fname: &str) -> RlstResult<DynamicArray<T, 2>> {
    let mut reader = open_file(fname).unwrap();
    let mm_info = parse_header(&mut reader).unwrap();

    if mm_info.format != MatrixFormat::Array {
        return Err(RlstError::IoError(
            "Matrix not in array format.".to_string(),
        ));
    }

    if mm_info.data_type == DataType::Integer {
        return Err(RlstError::IoError(
            "Integer matrices not supported.".to_string(),
        ));
    }

    if mm_info.symmetry != SymmetryType::General {
        return Err(RlstError::IoError(
            "Only matrices with Symmetry type `general` currently supported.".to_string(),
        ));
    }

    let mut nrows = 0;
    let mut ncols = 0;

    for line in reader.by_ref() {
        //while let Some(line) = reader.next() {
        let current_str = line.unwrap().to_string();
        if !current_str.starts_with('%') {
            let items = current_str
                .split_whitespace()
                .map(|elem| elem.to_string())
                .collect::<Vec<String>>();

            if items.len() != 2 {
                return Err(RlstError::IoError(
                    "Dimension line has unknown format.".to_string(),
                ));
            }

            nrows = items[0].parse().unwrap();
            ncols = items[1].parse().unwrap();

            break;
        }
    }

    let mut mat = rlst_dense::rlst_dynamic_array2!(T, [nrows, ncols]);
    let res = parse_array(&mut reader, mat.data_mut(), nrows * ncols);

    if let Err(e) = res {
        Err(e)
    } else {
        Ok(mat)
    }
}

/// Read a coordinate matrix in Matrix Market format.
///
/// Returns a [rlst_sparse::sparse::csr_mat::CsrMatrix] sparse matrix object representing
/// the data in the file.
/// Currently only `general` matrices are supported without special symmetry.
pub fn read_coordinate_mm<T: RlstScalar>(fname: &str) -> RlstResult<CsrMatrix<T>> {
    let mut reader = open_file(fname).unwrap();
    let mm_info = parse_header(&mut reader).unwrap();

    if mm_info.format != MatrixFormat::Coordinate {
        return Err(RlstError::IoError(
            "Matrix not in coordinate format.".to_string(),
        ));
    }

    if mm_info.data_type == DataType::Integer {
        return Err(RlstError::IoError(
            "Integer matrices not supported.".to_string(),
        ));
    }

    if mm_info.symmetry != SymmetryType::General {
        return Err(RlstError::IoError(
            "Only matrices with Symetry type `general` currently supported.".to_string(),
        ));
    }

    let mut nrows = 0;
    let mut ncols = 0;
    let mut nelems = 0;

    for line in reader.by_ref() {
        let current_str = line.unwrap().to_string();
        if !current_str.starts_with('%') {
            let items = current_str
                .split_whitespace()
                .map(|elem| elem.to_string())
                .collect::<Vec<String>>();

            if items.len() != 3 {
                return Err(RlstError::IoError(
                    "Dimension line has unknown format.".to_string(),
                ));
            }

            nrows = items[0].parse().unwrap();
            ncols = items[1].parse().unwrap();
            nelems = items[2].parse().unwrap();

            break;
        }
    }

    let mut rows = vec![0; nelems];
    let mut cols = vec![0; nelems];
    let mut data = vec![T::zero(); nelems];

    let res = parse_coordinate(&mut reader, &mut rows, &mut cols, &mut data, nelems);

    if let Err(e) = res {
        Err(e)
    } else {
        rows.iter_mut().for_each(|elem| *elem -= 1);
        cols.iter_mut().for_each(|elem| *elem -= 1);
        Ok(CsrMatrix::from_aij([nrows, ncols], &rows, &cols, &data).unwrap())
    }
}

/// Open a file and return the file handler.
fn open_file<P>(fname: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(fname)?;
    Ok(io::BufReader::new(file).lines())
}

/// Parse the header information.
fn parse_header(reader: &mut io::Lines<io::BufReader<File>>) -> RlstResult<MatrixMarketInfo> {
    if let Some(line) = reader.next() {
        let items = line
            .unwrap()
            .split_whitespace()
            .map(|elem| elem.to_string())
            .collect::<Vec<String>>();
        if items.len() != 5 || items[0] != "%%MatrixMarket" || items[1] != "matrix" {
            return Err(RlstError::IoError(
                "Header line has unknown format.".to_string(),
            ));
        }

        let format;
        if items[2] == "coordinate" {
            format = MatrixFormat::Coordinate;
        } else if items[2] == "array" {
            format = MatrixFormat::Array;
        } else {
            return Err(RlstError::IoError(
                "Header line has unknown format.".to_string(),
            ));
        }

        let data_type;
        if items[3] == "real" {
            data_type = DataType::Real;
        } else if items[3] == "complex" {
            data_type = DataType::Complex;
        } else if items[3] == "integer" {
            data_type = DataType::Integer;
        } else {
            return Err(RlstError::IoError(
                "Header line has unknown format.".to_string(),
            ));
        }

        let symmetry: SymmetryType;
        if items[4] == "general" {
            symmetry = SymmetryType::General;
        } else if items[4] == "symmetric" {
            symmetry = SymmetryType::Symmetric;
        } else if items[4] == "skew-symmetric" {
            symmetry = SymmetryType::Skew;
        } else {
            return Err(RlstError::IoError(
                "Header line has unknown format.".to_string(),
            ));
        }

        return Ok(MatrixMarketInfo {
            format,
            data_type,
            symmetry,
        });
    }
    Err(RlstError::IoError(
        "Failed to read header line.".to_string(),
    ))
}

/// Parse array information.
fn parse_array<T: RlstScalar>(
    reader: &mut io::Lines<io::BufReader<File>>,
    buf: &mut [T],
    nelems: usize,
) -> RlstResult<()> {
    let mut count = 0;

    for line in reader {
        let current_str = line.unwrap();
        let items = &current_str
            .split_whitespace()
            .map(|elem| elem.to_string())
            .collect::<Vec<String>>();

        if items.len() != 1 {
            return Err(RlstError::IoError(format!(
                "Failed to read data line: {}.",
                current_str,
            )));
        }

        let value = T::from_str_radix(&items[0], 10);

        match value {
            Ok(value) => {
                buf[count] = value;
                count += 1
            }
            Err(_) => {
                return Err(RlstError::IoError(format!(
                    "Failed to read data line: {}.",
                    current_str,
                )));
            }
        };
        if count == nelems {
            break;
        }
    }

    if count != nelems {
        return Err(RlstError::IoError(format!(
            "There were only {} data lines, expected {}",
            count, nelems
        )));
    }

    Ok(())
}

/// Parse coordinate information.
fn parse_coordinate<T: RlstScalar>(
    reader: &mut io::Lines<io::BufReader<File>>,
    rows: &mut [usize],
    cols: &mut [usize],
    data: &mut [T],
    nelems: usize,
) -> RlstResult<()> {
    let mut count = 0;

    for line in reader {
        let current_str = line.unwrap();
        let items = &current_str
            .split_whitespace()
            .map(|elem| elem.to_string())
            .collect::<Vec<String>>();

        if items.len() != 3 {
            return Err(RlstError::IoError(format!(
                "Failed to read data line: {}.",
                current_str,
            )));
        }

        let row = items[0].parse();
        let col = items[1].parse();
        let val = T::from_str_radix(&items[2], 10);

        match row {
            Ok(row) => {
                rows[count] = row;
            }
            Err(_) => {
                return Err(RlstError::IoError(format!(
                    "Failed to read data line: {}.",
                    current_str,
                )));
            }
        };

        match col {
            Ok(col) => {
                cols[count] = col;
            }
            Err(_) => {
                return Err(RlstError::IoError(format!(
                    "Failed to read data line: {}.",
                    current_str,
                )));
            }
        };

        match val {
            Ok(val) => {
                data[count] = val;
            }
            Err(_) => {
                return Err(RlstError::IoError(format!(
                    "Failed to read data line: {}.",
                    current_str,
                )));
            }
        };
        count += 1;
        if count == nelems {
            break;
        }
    }

    if count != nelems {
        return Err(RlstError::IoError(format!(
            "There were only {} data lines, expected {}",
            count, nelems
        )));
    }

    Ok(())
}
