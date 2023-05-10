//! Matrix market reader/writer
//!
//! The functions [read_array_mm] and [read_coordinate_mm] can read the Matrix Market array and coordinate
//! file formats. Currently, matrices with special symmetry properties are not supported. Input
//! files must use the `general` property. To write out matrices the functions [write_array_mm] and
//! [write_coordinate_mm] are provided.

use rlst_common::traits::RawAccessMut;
use rlst_common::types::Scalar;
use rlst_common::types::{RlstError, RlstResult};
use rlst_dense::matrix::MatrixD;
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
    const MMTYPE: &'static str;
}

impl MmIdentifier for f32 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for f64 {
    const MMTYPE: &'static str = "real";
}

impl MmIdentifier for rlst_common::types::c32 {
    const MMTYPE: &'static str = "complex";
}

impl MmIdentifier for rlst_common::types::c64 {
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
/// This function requires objects to implement the [rlst_common::traits::AijIterator] and
/// [rlst_common::traits::Shape] traits. Any object satisfying these traits can be written
/// out with this function.
pub fn write_coordinate_mm<
    T: Scalar + MmIdentifier,
    Mat: rlst_common::traits::AijIterator<T = T> + rlst_common::traits::Shape,
>(
    mat: &Mat,
    fname: &str,
) -> RlstResult<()> {
    let output = File::create(fname);

    let count = mat.iter_aij().count();

    if let Ok(mut output) = output {
        write!(
            output,
            "%%MatrixMarket matrix coordinate {} general\n",
            T::MMTYPE
        )
        .unwrap();
        write!(output, "%\n").unwrap();
        write!(output, "{} {} {}\n", mat.shape().0, mat.shape().1, count).unwrap();
        for (row, col, data) in mat.iter_aij() {
            write!(output, "{} {} {} \n", 1 + row, 1 + col, data).unwrap();
        }

        Ok(())
    } else {
        Err(RlstError::IoError(format!("Could not open file {}", fname)))
    }
}

/// Export a matrix in array format.
///
/// This function requires objects to implement the [rlst_common::traits::ColumnMajorIterator] and
/// [rlst_common::traits::Shape] traits. Any object satisfying these traits can be written
/// out with this function.
pub fn write_array_mm<
    T: Scalar + MmIdentifier,
    Mat: rlst_common::traits::ColumnMajorIterator<T = T> + rlst_common::traits::Shape,
>(
    mat: &Mat,
    fname: &str,
) -> RlstResult<()> {
    let output = File::create(fname);

    if let Ok(mut output) = output {
        write!(
            output,
            "%%MatrixMarket matrix array {} general\n",
            T::MMTYPE
        )
        .unwrap();
        write!(output, "%\n").unwrap();
        write!(output, "{} {}\n", mat.shape().0, mat.shape().1).unwrap();
        for value in mat.iter_col_major() {
            write!(output, "{}\n", value).unwrap();
        }

        Ok(())
    } else {
        Err(RlstError::IoError(format!("Could not open file {}", fname)))
    }
}

/// Read an array in matrix market format.
///
/// The function returns a [MatrixD] object representing the data in the file.
/// Currently only `general` matrices are supported without special symmetry.
pub fn read_array_mm<T: Scalar>(fname: &str) -> RlstResult<MatrixD<T>> {
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

    while let Some(line) = reader.next() {
        let current_str = line.unwrap().to_string();
        if !current_str.starts_with("%") {
            let items = current_str
                .split_whitespace()
                .map(|elem| elem.to_string())
                .collect::<Vec<String>>();

            if items.len() != 2 {
                return Err(RlstError::IoError(
                    "Dimension line has unknown format.".to_string(),
                ));
            }

            nrows = usize::from_str_radix(&items[0], 10).unwrap();
            ncols = usize::from_str_radix(&items[1], 10).unwrap();

            break;
        }
    }

    let mut mat = rlst_dense::rlst_mat!(T, (nrows, ncols));
    let res = parse_array(&mut reader, mat.data_mut(), nrows * ncols);

    if let Ok(_) = res {
        return Ok(mat);
    } else {
        return Err(res.unwrap_err());
    }
}

/// Read a coordinate matrix in Matrix Market format.
///
/// Returns a [rlst_sparse::sparse::csr_mat::CsrMatrix] sparse matrix object representing
/// the data in the file.
/// Currently only `general` matrices are supported without special symmetry.
pub fn read_coordinate_mm<T: Scalar>(fname: &str) -> RlstResult<CsrMatrix<T>> {
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

    while let Some(line) = reader.next() {
        let current_str = line.unwrap().to_string();
        if !current_str.starts_with("%") {
            let items = current_str
                .split_whitespace()
                .map(|elem| elem.to_string())
                .collect::<Vec<String>>();

            if items.len() != 3 {
                return Err(RlstError::IoError(
                    "Dimension line has unknown format.".to_string(),
                ));
            }

            nrows = usize::from_str_radix(&items[0], 10).unwrap();
            ncols = usize::from_str_radix(&items[1], 10).unwrap();
            nelems = usize::from_str_radix(&items[2], 10).unwrap();

            break;
        }
    }

    let mut rows = vec![0 as usize; nelems];
    let mut cols = vec![0 as usize; nelems];
    let mut data = vec![T::zero(); nelems];

    let res = parse_coordinate(&mut reader, &mut rows, &mut cols, &mut data, nelems);

    if let Ok(_) = res {
        rows.iter_mut().for_each(|elem| *elem = *elem - 1);
        cols.iter_mut().for_each(|elem| *elem = *elem - 1);
        return Ok(CsrMatrix::from_aij((nrows, ncols), &rows, &cols, &data).unwrap());
    } else {
        return Err(res.unwrap_err());
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
fn parse_array<T: Scalar>(
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
        return Err(RlstError::IoError(
            format!("There were only {} data lines, expected {}", count, nelems).to_string(),
        ));
    }

    Ok(())
}

/// Parse coordinate information.
fn parse_coordinate<T: Scalar>(
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

        let row = usize::from_str_radix(&items[0], 10);
        let col = usize::from_str_radix(&items[1], 10);
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
        return Err(RlstError::IoError(
            format!("There were only {} data lines, expected {}", count, nelems).to_string(),
        ));
    }

    Ok(())
}
