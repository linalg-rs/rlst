#define len_trim__(cad,len) ({                           integer _r=0,i;                           for(i=0; i<(len) && (cad)[i]; i++)                             if((cad)[i] != ' ') _r=i;                           _r+1; })
#define ceiling_(a) (myceil(*(a)))
#define myceil(a) (sizeof(a) == sizeof(float) ? ceilf(a) : ceil(a))
#include <math.h>
/*  -- translated by f2c (version 20200916).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#define __LAPACK_PRECISION_SINGLE
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;

/* > \brief \b STBTRS */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download STBTRS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/stbtrs.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/stbtrs.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/stbtrs.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE STBTRS( UPLO, TRANS, DIAG, N, KD, NRHS, AB, LDAB, B, */
/*                          LDB, INFO ) */

/*       CHARACTER          DIAG, TRANS, UPLO */
/*       INTEGER            INFO, KD, LDAB, LDB, N, NRHS */
/*       REAL               AB( LDAB, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > STBTRS solves a triangular system of the form */
/* > */
/* >    A * X = B  or  A**T * X = B, */
/* > */
/* > where A is a triangular band matrix of order N, and B is an */
/* > N-by NRHS matrix.  A check is made to verify that A is nonsingular. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  A is upper triangular; */
/* >          = 'L':  A is lower triangular. */
/* > \endverbatim */
/* > */
/* > \param[in] TRANS */
/* > \verbatim */
/* >          TRANS is CHARACTER*1 */
/* >          Specifies the form the system of equations: */
/* >          = 'N':  A * X = B  (No transpose) */
/* >          = 'T':  A**T * X = B  (Transpose) */
/* >          = 'C':  A**H * X = B  (Conjugate transpose = Transpose) */
/* > \endverbatim */
/* > */
/* > \param[in] DIAG */
/* > \verbatim */
/* >          DIAG is CHARACTER*1 */
/* >          = 'N':  A is non-unit triangular; */
/* >          = 'U':  A is unit triangular. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] KD */
/* > \verbatim */
/* >          KD is INTEGER */
/* >          The number of superdiagonals or subdiagonals of the */
/* >          triangular band matrix A.  KD >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrix B.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] AB */
/* > \verbatim */
/* >          AB is REAL array, dimension (LDAB,N) */
/* >          The upper or lower triangular band matrix A, stored in the */
/* >          first kd+1 rows of AB.  The j-th column of A is stored */
/* >          in the j-th column of the array AB as follows: */
/* >          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for f2cmax(1,j-kd)<=i<=j; */
/* >          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=f2cmin(n,j+kd). */
/* >          If DIAG = 'U', the diagonal elements of A are not referenced */
/* >          and are assumed to be 1. */
/* > \endverbatim */
/* > */
/* > \param[in] LDAB */
/* > \verbatim */
/* >          LDAB is INTEGER */
/* >          The leading dimension of the array AB.  LDAB >= KD+1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is REAL array, dimension (LDB,NRHS) */
/* >          On entry, the right hand side matrix B. */
/* >          On exit, if INFO = 0, the solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* >          > 0:  if INFO = i, the i-th diagonal element of A is zero, */
/* >                indicating that the matrix is singular and the */
/* >                solutions X have not been computed. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup realOTHERcomputational */

/*  ===================================================================== */
void  stbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, real *ab, integer *ldab, real *b, integer 
	*ldb, integer *info)
{
    /* System generated locals */
    integer ab_dim1, ab_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    integer j;
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  stbsv_(char *, char *, char *, integer *, 
	    integer *, real *, integer *, real *, integer *), xerbla_(char *, integer *);
    logical nounit;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    ab_dim1 = *ldab;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    nounit = lsame_(diag, "N");
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, 
	    "T") && ! lsame_(trans, "C")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*kd < 0) {
	*info = -5;
    } else if (*nrhs < 0) {
	*info = -6;
    } else if (*ldab < *kd + 1) {
	*info = -8;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STBTRS", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Check for singularity. */

    if (nounit) {
	if (upper) {
	    i__1 = *n;
	    for (*info = 1; *info <= i__1; ++(*info)) {
		if (ab[*kd + 1 + *info * ab_dim1] == 0.f) {
		    return;
		}
/* L10: */
	    }
	} else {
	    i__1 = *n;
	    for (*info = 1; *info <= i__1; ++(*info)) {
		if (ab[*info * ab_dim1 + 1] == 0.f) {
		    return;
		}
/* L20: */
	    }
	}
    }
    *info = 0;

/*     Solve A * X = B  or  A**T * X = B. */

    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {
	stbsv_(uplo, trans, diag, n, kd, &ab[ab_offset], ldab, &b[j * b_dim1 
		+ 1], &c__1);
/* L30: */
    }

    return;

/*     End of STBTRS */

} /* stbtrs_ */

