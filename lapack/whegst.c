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
static quadcomplex c_b2 = {.5,0.};
static integer c__1 = 1;
static integer c_n1 = -1;
static quadreal c_b18 = 1.;

/* > \brief \b ZHEGST */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZHEGST + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zhegst.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zhegst.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zhegst.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZHEGST( ITYPE, UPLO, N, A, LDA, B, LDB, INFO ) */

/*       CHARACTER          UPLO */
/*       INTEGER            INFO, ITYPE, LDA, LDB, N */
/*       COMPLEX*16         A( LDA, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZHEGST reduces a complex Hermitian-definite generalized */
/* > eigenproblem to standard form. */
/* > */
/* > If ITYPE = 1, the problem is A*x = lambda*B*x, */
/* > and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H) */
/* > */
/* > If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or */
/* > B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L. */
/* > */
/* > B must have been previously factorized as U**H*U or L*L**H by ZPOTRF. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] ITYPE */
/* > \verbatim */
/* >          ITYPE is INTEGER */
/* >          = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H); */
/* >          = 2 or 3: compute U*A*U**H or L**H*A*L. */
/* > \endverbatim */
/* > */
/* > \param[in] UPLO */
/* > \verbatim */
/* >          UPLO is CHARACTER*1 */
/* >          = 'U':  Upper triangle of A is stored and B is factored as */
/* >                  U**H*U; */
/* >          = 'L':  Lower triangle of A is stored and B is factored as */
/* >                  L*L**H. */
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
/* >          A is COMPLEX*16 array, dimension (LDA,N) */
/* >          On entry, the Hermitian matrix A.  If UPLO = 'U', the leading */
/* >          N-by-N upper triangular part of A contains the upper */
/* >          triangular part of the matrix A, and the strictly lower */
/* >          triangular part of A is not referenced.  If UPLO = 'L', the */
/* >          leading N-by-N lower triangular part of A contains the lower */
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
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX*16 array, dimension (LDB,N) */
/* >          The triangular factor from the Cholesky factorization of B, */
/* >          as returned by ZPOTRF. */
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

/* > \ingroup complex16HEcomputational */

/*  ===================================================================== */
void  whegst_(integer *itype, char *uplo, integer *n, 
	quadcomplex *a, integer *lda, quadcomplex *b, integer *ldb, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    quadcomplex z__1;

    /* Local variables */
    integer k, kb, nb;
    extern logical lsame_(char *, char *);
    extern void  whemm_(char *, char *, integer *, integer *, 
	    quadcomplex *, quadcomplex *, integer *, quadcomplex *, 
	    integer *, quadcomplex *, quadcomplex *, integer *);
    logical upper;
    extern void  wtrmm_(char *, char *, char *, char *, 
	    integer *, integer *, quadcomplex *, quadcomplex *, integer *,
	     quadcomplex *, integer *), 
	    wtrsm_(char *, char *, char *, char *, integer *, integer *, 
	    quadcomplex *, quadcomplex *, integer *, quadcomplex *, 
	    integer *), whegs2_(integer *, 
	    char *, integer *, quadcomplex *, integer *, quadcomplex *, 
	    integer *, integer *), wher2k_(char *, char *, integer *, 
	    integer *, quadcomplex *, quadcomplex *, integer *, 
	    quadcomplex *, integer *, quadreal *, quadcomplex *, 
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);


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
	xerbla_("ZHEGST", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "ZHEGST", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6, (
	    ftnlen)1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	whegs2_(itype, uplo, n, &a[a_offset], lda, &b[b_offset], ldb, info);
    } else {

/*        Use blocked code */

	if (*itype == 1) {
	    if (upper) {

/*              Compute inv(U**H)*A*inv(U) */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = f2cmin(i__3,nb);

/*                 Update the upper triangle of A(k:n,k:n) */

		    whegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			wtrsm_("Left", uplo, "Conjugate transpose", "Non-unit"
				, &kb, &i__3, &c_b1, &b[k + k * b_dim1], ldb, 
				&a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			z__1.r = -.5, z__1.i = -0.;
			whemm_("Left", uplo, &kb, &i__3, &z__1, &a[k + k * 
				a_dim1], lda, &b[k + (k + kb) * b_dim1], ldb, 
				&c_b1, &a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			z__1.r = -1., z__1.i = -0.;
			wher2k_(uplo, "Conjugate transpose", &i__3, &kb, &
				z__1, &a[k + (k + kb) * a_dim1], lda, &b[k + (
				k + kb) * b_dim1], ldb, &c_b18, &a[k + kb + (
				k + kb) * a_dim1], lda)
				;
			i__3 = *n - k - kb + 1;
			z__1.r = -.5, z__1.i = -0.;
			whemm_("Left", uplo, &kb, &i__3, &z__1, &a[k + k * 
				a_dim1], lda, &b[k + (k + kb) * b_dim1], ldb, 
				&c_b1, &a[k + (k + kb) * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			wtrsm_("Right", uplo, "No transpose", "Non-unit", &kb,
				 &i__3, &c_b1, &b[k + kb + (k + kb) * b_dim1],
				 ldb, &a[k + (k + kb) * a_dim1], lda);
		    }
/* L10: */
		}
	    } else {

/*              Compute inv(L)*A*inv(L**H) */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = f2cmin(i__3,nb);

/*                 Update the lower triangle of A(k:n,k:n) */

		    whegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
		    if (k + kb <= *n) {
			i__3 = *n - k - kb + 1;
			wtrsm_("Right", uplo, "Conjugate transpose", "Non-un"
				"it", &i__3, &kb, &c_b1, &b[k + k * b_dim1], 
				ldb, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			z__1.r = -.5, z__1.i = -0.;
			whemm_("Right", uplo, &i__3, &kb, &z__1, &a[k + k * 
				a_dim1], lda, &b[k + kb + k * b_dim1], ldb, &
				c_b1, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			z__1.r = -1., z__1.i = -0.;
			wher2k_(uplo, "No transpose", &i__3, &kb, &z__1, &a[k 
				+ kb + k * a_dim1], lda, &b[k + kb + k * 
				b_dim1], ldb, &c_b18, &a[k + kb + (k + kb) * 
				a_dim1], lda);
			i__3 = *n - k - kb + 1;
			z__1.r = -.5, z__1.i = -0.;
			whemm_("Right", uplo, &i__3, &kb, &z__1, &a[k + k * 
				a_dim1], lda, &b[k + kb + k * b_dim1], ldb, &
				c_b1, &a[k + kb + k * a_dim1], lda);
			i__3 = *n - k - kb + 1;
			wtrsm_("Left", uplo, "No transpose", "Non-unit", &
				i__3, &kb, &c_b1, &b[k + kb + (k + kb) * 
				b_dim1], ldb, &a[k + kb + k * a_dim1], lda);
		    }
/* L20: */
		}
	    }
	} else {
	    if (upper) {

/*              Compute U*A*U**H */

		i__1 = *n;
		i__2 = nb;
		for (k = 1; i__2 < 0 ? k >= i__1 : k <= i__1; k += i__2) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = f2cmin(i__3,nb);

/*                 Update the upper triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    wtrmm_("Left", uplo, "No transpose", "Non-unit", &i__3, &
			    kb, &c_b1, &b[b_offset], ldb, &a[k * a_dim1 + 1], 
			    lda);
		    i__3 = k - 1;
		    whemm_("Right", uplo, &i__3, &kb, &c_b2, &a[k + k * 
			    a_dim1], lda, &b[k * b_dim1 + 1], ldb, &c_b1, &a[
			    k * a_dim1 + 1], lda);
		    i__3 = k - 1;
		    wher2k_(uplo, "No transpose", &i__3, &kb, &c_b1, &a[k * 
			    a_dim1 + 1], lda, &b[k * b_dim1 + 1], ldb, &c_b18,
			     &a[a_offset], lda);
		    i__3 = k - 1;
		    whemm_("Right", uplo, &i__3, &kb, &c_b2, &a[k + k * 
			    a_dim1], lda, &b[k * b_dim1 + 1], ldb, &c_b1, &a[
			    k * a_dim1 + 1], lda);
		    i__3 = k - 1;
		    wtrmm_("Right", uplo, "Conjugate transpose", "Non-unit", &
			    i__3, &kb, &c_b1, &b[k + k * b_dim1], ldb, &a[k * 
			    a_dim1 + 1], lda);
		    whegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
/* L30: */
		}
	    } else {

/*              Compute L**H*A*L */

		i__2 = *n;
		i__1 = nb;
		for (k = 1; i__1 < 0 ? k >= i__2 : k <= i__2; k += i__1) {
/* Computing MIN */
		    i__3 = *n - k + 1;
		    kb = f2cmin(i__3,nb);

/*                 Update the lower triangle of A(1:k+kb-1,1:k+kb-1) */

		    i__3 = k - 1;
		    wtrmm_("Right", uplo, "No transpose", "Non-unit", &kb, &
			    i__3, &c_b1, &b[b_offset], ldb, &a[k + a_dim1], 
			    lda);
		    i__3 = k - 1;
		    whemm_("Left", uplo, &kb, &i__3, &c_b2, &a[k + k * a_dim1]
			    , lda, &b[k + b_dim1], ldb, &c_b1, &a[k + a_dim1],
			     lda);
		    i__3 = k - 1;
		    wher2k_(uplo, "Conjugate transpose", &i__3, &kb, &c_b1, &
			    a[k + a_dim1], lda, &b[k + b_dim1], ldb, &c_b18, &
			    a[a_offset], lda);
		    i__3 = k - 1;
		    whemm_("Left", uplo, &kb, &i__3, &c_b2, &a[k + k * a_dim1]
			    , lda, &b[k + b_dim1], ldb, &c_b1, &a[k + a_dim1],
			     lda);
		    i__3 = k - 1;
		    wtrmm_("Left", uplo, "Conjugate transpose", "Non-unit", &
			    kb, &i__3, &c_b1, &b[k + k * b_dim1], ldb, &a[k + 
			    a_dim1], lda);
		    whegs2_(itype, uplo, &kb, &a[k + k * a_dim1], lda, &b[k + 
			    k * b_dim1], ldb, info);
/* L40: */
		}
	    }
	}
    }
    return;

/*     End of ZHEGST */

} /* whegst_ */

