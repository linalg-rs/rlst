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

static real c_b6 = -1.f;
static integer c__1 = 1;
static real c_b27 = 1.f;

/* > \brief \b SSYGS2 reduces a symmetric definite generalized eigenproblem to standard form, using the factor
ization results obtained from spotrf (unblocked algorithm). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SSYGS2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ssygs2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ssygs2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ssygs2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SSYGS2( ITYPE, UPLO, N, A, LDA, B, LDB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, ITYPE, LDA, LDB, N */
/*       REAL               A( LDA, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SSYGS2 reduces a real symmetric-definite generalized eigenproblem */
/* > to standard form. */
/* > */
/* > If ITYPE = 1, the problem is A*x = lambda*B*x, */
/* > and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T) */
/* > */
/* > If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/* > B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T *A*L. */
/* > */
/* > B must have been previously factorized as U**T *U or L*L**T by SPOTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] ITYPE */
/* > \verbatim */
/* >          ITYPE is INTEGER */
/* >          = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T); */
/* >          = 2 or 3: compute U*A*U**T or L**T *A*L. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          Specifies whether the upper or lower triangular part of the */
/* >          symmetric matrix A is stored, and how B has been factorized. */
/* >          = 'U':  Upper triangular */
/* >          = 'L':  Lower triangular */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrices A and B.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is REAL array, dimension (LDA,N) */
/* >          On entry, the symmetric matrix A.  If UPLO = 'U', the leading */
/* >          n by n upper triangular part of A contains the upper */
/* >          triangular part of the matrix A, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading n by n lower triangular part of A contains the lower */
/* >          triangular part of the matrix A, and the strictly upper */
/* >          triangular part of A is not referenced. */
/* > */
/* >          On exit, if INFO = 0, the transformed matrix, stored in the */
/* >          same format as A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is REAL array, dimension (LDB,N) */
/* >          The triangular factor from the Cholesky factorization of B, */
/* >          as returned by SPOTRF. */
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
/* >          = 0:  successful exit. */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup realSYcomputational */

/*  ===================================================================== */
void  ssygs2_(integer *itype, char *uplo, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;
    real r__1;

    /* Local variables */
    integer k;
    real ct, akk, bkk;
    extern void  ssyr2_(char *, integer *, real *, real *, 
	    integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern void  sscal_(integer *, real *, real *, integer *);
    logical upper;
    extern void  saxpy_(integer *, real *, real *, integer *, 
	    real *, integer *), strmv_(char *, char *, char *, integer *, 
	    real *, integer *, real *, integer *), 
	    strsv_(char *, char *, char *, integer *, real *, integer *, real 
	    *, integer *), xerbla_(char *, integer *);


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
    if (*itype < 1 || *itype > 3) {
	*info = -1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < f2cmax(1,*n)) {
	*info = -5;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SSYGS2", &i__1);
	return;
    }

    if (*itype == 1) {
	if (upper) {

/*           Compute inv(U**T)*A*inv(U) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(k:n,k:n) */

		akk = a[k + k * a_dim1];
		bkk = b[k + k * b_dim1];
/* Computing 2nd power */
		r__1 = bkk;
		akk /= r__1 * r__1;
		a[k + k * a_dim1] = akk;
		if (k < *n) {
		    i__2 = *n - k;
		    r__1 = 1.f / bkk;
		    sscal_(&i__2, &r__1, &a[k + (k + 1) * a_dim1], lda);
		    ct = akk * -.5f;
		    i__2 = *n - k;
		    saxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    ssyr2_(uplo, &i__2, &c_b6, &a[k + (k + 1) * a_dim1], lda, 
			    &b[k + (k + 1) * b_dim1], ldb, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    saxpy_(&i__2, &ct, &b[k + (k + 1) * b_dim1], ldb, &a[k + (
			    k + 1) * a_dim1], lda);
		    i__2 = *n - k;
		    strsv_(uplo, "Transpose", "Non-unit", &i__2, &b[k + 1 + (
			    k + 1) * b_dim1], ldb, &a[k + (k + 1) * a_dim1], 
			    lda);
		}
/* L10: */
	    }
	} else {

/*           Compute inv(L)*A*inv(L**T) */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(k:n,k:n) */

		akk = a[k + k * a_dim1];
		bkk = b[k + k * b_dim1];
/* Computing 2nd power */
		r__1 = bkk;
		akk /= r__1 * r__1;
		a[k + k * a_dim1] = akk;
		if (k < *n) {
		    i__2 = *n - k;
		    r__1 = 1.f / bkk;
		    sscal_(&i__2, &r__1, &a[k + 1 + k * a_dim1], &c__1);
		    ct = akk * -.5f;
		    i__2 = *n - k;
		    saxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    ssyr2_(uplo, &i__2, &c_b6, &a[k + 1 + k * a_dim1], &c__1, 
			    &b[k + 1 + k * b_dim1], &c__1, &a[k + 1 + (k + 1) 
			    * a_dim1], lda);
		    i__2 = *n - k;
		    saxpy_(&i__2, &ct, &b[k + 1 + k * b_dim1], &c__1, &a[k + 
			    1 + k * a_dim1], &c__1);
		    i__2 = *n - k;
		    strsv_(uplo, "No transpose", "Non-unit", &i__2, &b[k + 1 
			    + (k + 1) * b_dim1], ldb, &a[k + 1 + k * a_dim1], 
			    &c__1);
		}
/* L20: */
	    }
	}
    } else {
	if (upper) {

/*           Compute U*A*U**T */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the upper triangle of A(1:k,1:k) */

		akk = a[k + k * a_dim1];
		bkk = b[k + k * b_dim1];
		i__2 = k - 1;
		strmv_(uplo, "No transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a[k * a_dim1 + 1], &c__1);
		ct = akk * .5f;
		i__2 = k - 1;
		saxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		ssyr2_(uplo, &i__2, &c_b27, &a[k * a_dim1 + 1], &c__1, &b[k * 
			b_dim1 + 1], &c__1, &a[a_offset], lda);
		i__2 = k - 1;
		saxpy_(&i__2, &ct, &b[k * b_dim1 + 1], &c__1, &a[k * a_dim1 + 
			1], &c__1);
		i__2 = k - 1;
		sscal_(&i__2, &bkk, &a[k * a_dim1 + 1], &c__1);
/* Computing 2nd power */
		r__1 = bkk;
		a[k + k * a_dim1] = akk * (r__1 * r__1);
/* L30: */
	    }
	} else {

/*           Compute L**T *A*L */

	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {

/*              Update the lower triangle of A(1:k,1:k) */

		akk = a[k + k * a_dim1];
		bkk = b[k + k * b_dim1];
		i__2 = k - 1;
		strmv_(uplo, "Transpose", "Non-unit", &i__2, &b[b_offset], 
			ldb, &a[k + a_dim1], lda);
		ct = akk * .5f;
		i__2 = k - 1;
		saxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		ssyr2_(uplo, &i__2, &c_b27, &a[k + a_dim1], lda, &b[k + 
			b_dim1], ldb, &a[a_offset], lda);
		i__2 = k - 1;
		saxpy_(&i__2, &ct, &b[k + b_dim1], ldb, &a[k + a_dim1], lda);
		i__2 = k - 1;
		sscal_(&i__2, &bkk, &a[k + a_dim1], lda);
/* Computing 2nd power */
		r__1 = bkk;
		a[k + k * a_dim1] = akk * (r__1 * r__1);
/* L40: */
	    }
	}
    }
    return;

/*     End of SSYGS2 */

} /* ssygs2_ */

