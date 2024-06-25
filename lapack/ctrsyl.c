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

/* > \brief \b CTRSYL */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CTRSYL + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ctrsyl.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ctrsyl.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ctrsyl.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CTRSYL( TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C, */
/*                          LDC, SCALE, INFO ) */

/*       CHARACTER          TRANA, TRANB */
/*       INTEGER            INFO, ISGN, LDA, LDB, LDC, M, N */
/*       REAL               SCALE */
/*       COMPLEX            A( LDA, * ), B( LDB, * ), C( LDC, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CTRSYL solves the complex Sylvester matrix equation: */
/* > */
/* >    op(A)*X + X*op(B) = scale*C or */
/* >    op(A)*X - X*op(B) = scale*C, */
/* > */
/* > where op(A) = A or A**H, and A and B are both upper triangular. A is */
/* > M-by-M and B is N-by-N; the right hand side C and the solution X are */
/* > M-by-N; and scale is an output scale factor, set <= 1 to avoid */
/* > overflow in X. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] TRANA */
/* > \verbatim */
/* >          TRANA is CHARACTER*1 */
/* >          Specifies the option op(A): */
/* >          = 'N': op(A) = A    (No transpose) */
/* >          = 'C': op(A) = A**H (Conjugate transpose) */
/* > \endverbatim */
/* > */
/* > \param[in] TRANB */
/* > \verbatim */
/* >          TRANB is CHARACTER*1 */
/* >          Specifies the option op(B): */
/* >          = 'N': op(B) = B    (No transpose) */
/* >          = 'C': op(B) = B**H (Conjugate transpose) */
/* > \endverbatim */
/* > */
/* > \param[in] ISGN */
/* > \verbatim */
/* >          ISGN is INTEGER */
/* >          Specifies the sign in the equation: */
/* >          = +1: solve op(A)*X + X*op(B) = scale*C */
/* >          = -1: solve op(A)*X - X*op(B) = scale*C */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The order of the matrix A, and the number of rows in the */
/* >          matrices X and C. M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix B, and the number of columns in the */
/* >          matrices X and C. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,M) */
/* >          The upper triangular matrix A. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A. LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,N) */
/* >          The upper triangular matrix B. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B. LDB >= f2cmax(1,N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is COMPLEX array, dimension (LDC,N) */
/* >          On entry, the M-by-N right hand side matrix C. */
/* >          On exit, C is overwritten by the solution matrix X. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C. LDC >= f2cmax(1,M) */
/* > \endverbatim */
/* > */
/* > \param[out] SCALE */
/* > \verbatim */
/* >          SCALE is REAL */
/* >          The scale factor, scale, set <= 1 to avoid overflow in X. */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0: successful exit */
/* >          < 0: if INFO = -i, the i-th argument had an illegal value */
/* >          = 1: A and B have common or very close eigenvalues; perturbed */
/* >               values were used to solve the equation (but the matrices */
/* >               A and B are unchanged). */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexSYcomputational */

/*  ===================================================================== */
void  ctrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, complex *a, integer *lda, complex *b, integer *ldb, 
	complex *c__, integer *ldc, real *scale, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
	    i__3, i__4;
    real r__1, r__2;
    complex q__1, q__2, q__3, q__4;

    /* Local variables */
    integer j, k, l;
    complex a11;
    real db;
    complex x11;
    real da11;
    complex vec;
    real dum[1], eps, sgn, smin;
    complex suml, sumr;
    extern /* Complex */ VOID cdotc_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Complex */ VOID cdotu_(complex *, integer *, complex *, integer 
	    *, complex *, integer *);
    extern void  slabad_(real *, real *);
    extern real clange_(char *, integer *, integer *, complex *, integer *, 
	    real *);
    extern /* Complex */ VOID cladiv_(complex *, complex *, complex *);
    real scaloc;
    extern real slamch_(char *);
    extern void  csscal_(integer *, real *, complex *, integer 
	    *), xerbla_(char *, integer *);
    real bignum;
    logical notrna, notrnb;
    real smlnum;


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


/*     Decode and Test input parameters */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    notrna = lsame_(trana, "N");
    notrnb = lsame_(tranb, "N");

    *info = 0;
    if (! notrna && ! lsame_(trana, "C")) {
	*info = -1;
    } else if (! notrnb && ! lsame_(tranb, "C")) {
	*info = -2;
    } else if (*isgn != 1 && *isgn != -1) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -7;
    } else if (*ldb < f2cmax(1,*n)) {
	*info = -9;
    } else if (*ldc < f2cmax(1,*m)) {
	*info = -11;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTRSYL", &i__1);
	return;
    }

/*     Quick return if possible */

    *scale = 1.f;
    if (*m == 0 || *n == 0) {
	return;
    }

/*     Set constants to control overflow */

    eps = slamch_("P");
    smlnum = slamch_("S");
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);
    smlnum = smlnum * (real) (*m * *n) / eps;
    bignum = 1.f / smlnum;
/* Computing MAX */
    r__1 = smlnum, r__2 = eps * clange_("M", m, m, &a[a_offset], lda, dum), r__1 = f2cmax(r__1,r__2), r__2 = eps * clange_("M", n, n, 
	    &b[b_offset], ldb, dum);
    smin = f2cmax(r__1,r__2);
    sgn = (real) (*isgn);

    if (notrna && notrnb) {

/*        Solve    A*X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-left corner column by column by */

/*            A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    M                        L-1 */
/*          R(K,L) = SUM [A(K,I)*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]. */
/*                  I=K+1                      J=1 */

	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    for (k = *m; k >= 1; --k) {

		i__2 = *m - k;
/* Computing MIN */
		i__3 = k + 1;
/* Computing MIN */
		i__4 = k + 1;
		cdotu_(&q__1, &i__2, &a[k + f2cmin(i__3,*m) * a_dim1], lda, &c__[
			f2cmin(i__4,*m) + l * c_dim1], &c__1);
		suml.r = q__1.r, suml.i = q__1.i;
		i__2 = l - 1;
		cdotu_(&q__1, &i__2, &c__[k + c_dim1], ldc, &b[l * b_dim1 + 1]
			, &c__1);
		sumr.r = q__1.r, sumr.i = q__1.i;
		i__2 = k + l * c_dim1;
		q__3.r = sgn * sumr.r, q__3.i = sgn * sumr.i;
		q__2.r = suml.r + q__3.r, q__2.i = suml.i + q__3.i;
		q__1.r = c__[i__2].r - q__2.r, q__1.i = c__[i__2].i - q__2.i;
		vec.r = q__1.r, vec.i = q__1.i;

		scaloc = 1.f;
		i__2 = k + k * a_dim1;
		i__3 = l + l * b_dim1;
		q__2.r = sgn * b[i__3].r, q__2.i = sgn * b[i__3].i;
		q__1.r = a[i__2].r + q__2.r, q__1.i = a[i__2].i + q__2.i;
		a11.r = q__1.r, a11.i = q__1.i;
		da11 = (r__1 = a11.r, abs(r__1)) + (r__2 = r_imag(&a11), abs(
			r__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.f;
		    da11 = smin;
		    *info = 1;
		}
		db = (r__1 = vec.r, abs(r__1)) + (r__2 = r_imag(&vec), abs(
			r__2));
		if (da11 < 1.f && db > 1.f) {
		    if (db > bignum * da11) {
			scaloc = 1.f / db;
		    }
		}
		q__3.r = scaloc, q__3.i = 0.f;
		q__2.r = vec.r * q__3.r - vec.i * q__3.i, q__2.i = vec.r * 
			q__3.i + vec.i * q__3.r;
		cladiv_(&q__1, &q__2, &a11);
		x11.r = q__1.r, x11.i = q__1.i;

		if (scaloc != 1.f) {
		    i__2 = *n;
		    for (j = 1; j <= i__2; ++j) {
			csscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L10: */
		    }
		    *scale *= scaloc;
		}
		i__2 = k + l * c_dim1;
		c__[i__2].r = x11.r, c__[i__2].i = x11.i;

/* L20: */
	    }
/* L30: */
	}

    } else if (! notrna && notrnb) {

/*        Solve    A**H *X + ISGN*X*B = scale*C. */

/*        The (K,L)th block of X is determined starting from */
/*        upper-left corner column by column by */

/*            A**H(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                   K-1                           L-1 */
/*          R(K,L) = SUM [A**H(I,K)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)] */
/*                   I=1                           J=1 */

	i__1 = *n;
	for (l = 1; l <= i__1; ++l) {
	    i__2 = *m;
	    for (k = 1; k <= i__2; ++k) {

		i__3 = k - 1;
		cdotc_(&q__1, &i__3, &a[k * a_dim1 + 1], &c__1, &c__[l * 
			c_dim1 + 1], &c__1);
		suml.r = q__1.r, suml.i = q__1.i;
		i__3 = l - 1;
		cdotu_(&q__1, &i__3, &c__[k + c_dim1], ldc, &b[l * b_dim1 + 1]
			, &c__1);
		sumr.r = q__1.r, sumr.i = q__1.i;
		i__3 = k + l * c_dim1;
		q__3.r = sgn * sumr.r, q__3.i = sgn * sumr.i;
		q__2.r = suml.r + q__3.r, q__2.i = suml.i + q__3.i;
		q__1.r = c__[i__3].r - q__2.r, q__1.i = c__[i__3].i - q__2.i;
		vec.r = q__1.r, vec.i = q__1.i;

		scaloc = 1.f;
		r_cnjg(&q__2, &a[k + k * a_dim1]);
		i__3 = l + l * b_dim1;
		q__3.r = sgn * b[i__3].r, q__3.i = sgn * b[i__3].i;
		q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
		a11.r = q__1.r, a11.i = q__1.i;
		da11 = (r__1 = a11.r, abs(r__1)) + (r__2 = r_imag(&a11), abs(
			r__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.f;
		    da11 = smin;
		    *info = 1;
		}
		db = (r__1 = vec.r, abs(r__1)) + (r__2 = r_imag(&vec), abs(
			r__2));
		if (da11 < 1.f && db > 1.f) {
		    if (db > bignum * da11) {
			scaloc = 1.f / db;
		    }
		}

		q__3.r = scaloc, q__3.i = 0.f;
		q__2.r = vec.r * q__3.r - vec.i * q__3.i, q__2.i = vec.r * 
			q__3.i + vec.i * q__3.r;
		cladiv_(&q__1, &q__2, &a11);
		x11.r = q__1.r, x11.i = q__1.i;

		if (scaloc != 1.f) {
		    i__3 = *n;
		    for (j = 1; j <= i__3; ++j) {
			csscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L40: */
		    }
		    *scale *= scaloc;
		}
		i__3 = k + l * c_dim1;
		c__[i__3].r = x11.r, c__[i__3].i = x11.i;

/* L50: */
	    }
/* L60: */
	}

    } else if (! notrna && ! notrnb) {

/*        Solve    A**H*X + ISGN*X*B**H = C. */

/*        The (K,L)th block of X is determined starting from */
/*        upper-right corner column by column by */

/*            A**H(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    K-1 */
/*           R(K,L) = SUM [A**H(I,K)*X(I,L)] + */
/*                    I=1 */
/*                           N */
/*                     ISGN*SUM [X(K,J)*B**H(L,J)]. */
/*                          J=L+1 */

	for (l = *n; l >= 1; --l) {
	    i__1 = *m;
	    for (k = 1; k <= i__1; ++k) {

		i__2 = k - 1;
		cdotc_(&q__1, &i__2, &a[k * a_dim1 + 1], &c__1, &c__[l * 
			c_dim1 + 1], &c__1);
		suml.r = q__1.r, suml.i = q__1.i;
		i__2 = *n - l;
/* Computing MIN */
		i__3 = l + 1;
/* Computing MIN */
		i__4 = l + 1;
		cdotc_(&q__1, &i__2, &c__[k + f2cmin(i__3,*n) * c_dim1], ldc, &b[
			l + f2cmin(i__4,*n) * b_dim1], ldb);
		sumr.r = q__1.r, sumr.i = q__1.i;
		i__2 = k + l * c_dim1;
		r_cnjg(&q__4, &sumr);
		q__3.r = sgn * q__4.r, q__3.i = sgn * q__4.i;
		q__2.r = suml.r + q__3.r, q__2.i = suml.i + q__3.i;
		q__1.r = c__[i__2].r - q__2.r, q__1.i = c__[i__2].i - q__2.i;
		vec.r = q__1.r, vec.i = q__1.i;

		scaloc = 1.f;
		i__2 = k + k * a_dim1;
		i__3 = l + l * b_dim1;
		q__3.r = sgn * b[i__3].r, q__3.i = sgn * b[i__3].i;
		q__2.r = a[i__2].r + q__3.r, q__2.i = a[i__2].i + q__3.i;
		r_cnjg(&q__1, &q__2);
		a11.r = q__1.r, a11.i = q__1.i;
		da11 = (r__1 = a11.r, abs(r__1)) + (r__2 = r_imag(&a11), abs(
			r__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.f;
		    da11 = smin;
		    *info = 1;
		}
		db = (r__1 = vec.r, abs(r__1)) + (r__2 = r_imag(&vec), abs(
			r__2));
		if (da11 < 1.f && db > 1.f) {
		    if (db > bignum * da11) {
			scaloc = 1.f / db;
		    }
		}

		q__3.r = scaloc, q__3.i = 0.f;
		q__2.r = vec.r * q__3.r - vec.i * q__3.i, q__2.i = vec.r * 
			q__3.i + vec.i * q__3.r;
		cladiv_(&q__1, &q__2, &a11);
		x11.r = q__1.r, x11.i = q__1.i;

		if (scaloc != 1.f) {
		    i__2 = *n;
		    for (j = 1; j <= i__2; ++j) {
			csscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L70: */
		    }
		    *scale *= scaloc;
		}
		i__2 = k + l * c_dim1;
		c__[i__2].r = x11.r, c__[i__2].i = x11.i;

/* L80: */
	    }
/* L90: */
	}

    } else if (notrna && ! notrnb) {

/*        Solve    A*X + ISGN*X*B**H = C. */

/*        The (K,L)th block of X is determined starting from */
/*        bottom-left corner column by column by */

/*           A(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L) */

/*        Where */
/*                    M                          N */
/*          R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B**H(L,J)] */
/*                  I=K+1                      J=L+1 */

	for (l = *n; l >= 1; --l) {
	    for (k = *m; k >= 1; --k) {

		i__1 = *m - k;
/* Computing MIN */
		i__2 = k + 1;
/* Computing MIN */
		i__3 = k + 1;
		cdotu_(&q__1, &i__1, &a[k + f2cmin(i__2,*m) * a_dim1], lda, &c__[
			f2cmin(i__3,*m) + l * c_dim1], &c__1);
		suml.r = q__1.r, suml.i = q__1.i;
		i__1 = *n - l;
/* Computing MIN */
		i__2 = l + 1;
/* Computing MIN */
		i__3 = l + 1;
		cdotc_(&q__1, &i__1, &c__[k + f2cmin(i__2,*n) * c_dim1], ldc, &b[
			l + f2cmin(i__3,*n) * b_dim1], ldb);
		sumr.r = q__1.r, sumr.i = q__1.i;
		i__1 = k + l * c_dim1;
		r_cnjg(&q__4, &sumr);
		q__3.r = sgn * q__4.r, q__3.i = sgn * q__4.i;
		q__2.r = suml.r + q__3.r, q__2.i = suml.i + q__3.i;
		q__1.r = c__[i__1].r - q__2.r, q__1.i = c__[i__1].i - q__2.i;
		vec.r = q__1.r, vec.i = q__1.i;

		scaloc = 1.f;
		i__1 = k + k * a_dim1;
		r_cnjg(&q__3, &b[l + l * b_dim1]);
		q__2.r = sgn * q__3.r, q__2.i = sgn * q__3.i;
		q__1.r = a[i__1].r + q__2.r, q__1.i = a[i__1].i + q__2.i;
		a11.r = q__1.r, a11.i = q__1.i;
		da11 = (r__1 = a11.r, abs(r__1)) + (r__2 = r_imag(&a11), abs(
			r__2));
		if (da11 <= smin) {
		    a11.r = smin, a11.i = 0.f;
		    da11 = smin;
		    *info = 1;
		}
		db = (r__1 = vec.r, abs(r__1)) + (r__2 = r_imag(&vec), abs(
			r__2));
		if (da11 < 1.f && db > 1.f) {
		    if (db > bignum * da11) {
			scaloc = 1.f / db;
		    }
		}

		q__3.r = scaloc, q__3.i = 0.f;
		q__2.r = vec.r * q__3.r - vec.i * q__3.i, q__2.i = vec.r * 
			q__3.i + vec.i * q__3.r;
		cladiv_(&q__1, &q__2, &a11);
		x11.r = q__1.r, x11.i = q__1.i;

		if (scaloc != 1.f) {
		    i__1 = *n;
		    for (j = 1; j <= i__1; ++j) {
			csscal_(m, &scaloc, &c__[j * c_dim1 + 1], &c__1);
/* L100: */
		    }
		    *scale *= scaloc;
		}
		i__1 = k + l * c_dim1;
		c__[i__1].r = x11.r, c__[i__1].i = x11.i;

/* L110: */
	    }
/* L120: */
	}

    }

    return;

/*     End of CTRSYL */

} /* ctrsyl_ */

