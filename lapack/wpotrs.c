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

#define __LAPACK_PRECISION_QUAD
#include "f2c.h"

/* Table of constant values */

static quadcomplex c_b1 = {1.,0.};

/* > \brief \b ZPOTRS */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZPOTRS + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zpotrs.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zpotrs.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zpotrs.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, LDA, LDB, N, NRHS */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZPOTRS solves a system of linear equations A*X = B with a Hermitian */
/* > positive definite matrix A using the Cholesky factorization */
/* > A = U**H * U or A = L * L**H computed by ZPOTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangle of A is stored; */
/* >          = 'L':  Lower triangle of A is stored. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] NRHS */
/* > \verbatim */
/* >          NRHS is INTEGER */
/* >          The number of right hand sides, i.e., the number of columns */
/* >          of the matrix B.  NRHS >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          The triangular factor U or L from the Cholesky factorization */
/* >          A = U**H * U or A = L * L**H, as computed by ZPOTRF. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,NRHS) */
/* >          On entry, the right hand side matrix B. */
/* >          On exit, the solution matrix X. */
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
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16POcomputational */

/*  ===================================================================== */
void  wpotrs_(char *uplo, integer *n, integer *nrhs, 
	quadcomplex *a, integer *lda, quadcomplex *b, integer *ldb, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1;

    /* Local variables */
    extern logical lsame_(char *, char *);
    logical upper;
    extern void  wtrsm_(char *, char *, char *, char *, 
	    integer *, integer *, quadcomplex *, quadcomplex *, integer *,
	     quadcomplex *, integer *), 
	    xerbla_(char *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("ZPOTRS", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return;
    }

    if (upper) {

/*        Solve A*X = B where A = U**H *U. */

/*        Solve U**H *X = B, overwriting B with X. */

	wtrsm_("Left", "Upper", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b1, &a[a_offset], lda, &b[b_offset], ldb);

/*        Solve U*X = B, overwriting B with X. */

	wtrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b1, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {

/*        Solve A*X = B where A = L*L**H. */

/*        Solve L*X = B, overwriting B with X. */

	wtrsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b1, &
		a[a_offset], lda, &b[b_offset], ldb);

/*        Solve L**H *X = B, overwriting B with X. */

	wtrsm_("Left", "Lower", "Conjugate transpose", "Non-unit", n, nrhs, &
		c_b1, &a[a_offset], lda, &b[b_offset], ldb);
    }

    return;

/*     End of ZPOTRS */

} /* wpotrs_ */

