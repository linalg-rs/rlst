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

static integer c__1 = 1;
static quadcomplex c_b13 = {1.,0.};
static integer c_n1 = -1;

/* > \brief \b ZGESC2 solves a system of linear equations using the LU factorization with complete pivoting co
mputed by sgetc2. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGESC2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgesc2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgesc2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgesc2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGESC2( N, A, LDA, RHS, IPIV, JPIV, SCALE ) */

/*       INTEGER            LDA, N */
/*       DOUBLE PRECISION   SCALE */
/*       INTEGER            IPIV( * ), JPIV( * ) */
/*       COMPLEX*16         A( LDA, * ), RHS( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGESC2 solves a system of linear equations */
/* > */
/* >           A * X = scale* RHS */
/* > */
/* > with a general N-by-N matrix A using the LU factorization with */
/* > complete pivoting computed by ZGETC2. */
/* > */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A. */
/* > \endverbatim */
/* > */
/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA, N) */
/* >          On entry, the  LU part of the factorization of the n-by-n */
/* >          matrix A computed by ZGETC2:  A = P * L * U * Q */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1, N). */
/* > \endverbatim */
/* > */
/* > \param[in,out] RHS */
/* > \verbatim */
/* >          RHS is COMPLEX*16 array, dimension N. */
/* >          On entry, the right hand side vector b. */
/* >          On exit, the solution vector X. */
/* > \endverbatim */
/* > */
/* > \param[in] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= i <= N, row i of the */
/* >          matrix has been interchanged with row IPIV(i). */
/* > \endverbatim */
/* > */
/* > \param[in] JPIV */
/* > \verbatim */
/* >          JPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= j <= N, column j of the */
/* >          matrix has been interchanged with column JPIV(j). */
/* > \endverbatim */
/* > */
/* > \param[out] SCALE */
/* > \verbatim */
/* >          SCALE is DOUBLE PRECISION */
/* >           On exit, SCALE contains the scale factor. SCALE is chosen */
/* >           0 <= SCALE <= 1 to prevent owerflow in the solution. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date November 2017 */

/* > \ingroup complex16GEauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/* >     Umea University, S-901 87 Umea, Sweden. */

/*  ===================================================================== */
void  wgesc2_(integer *n, quadcomplex *a, integer *lda, 
	quadcomplex *rhs, integer *ipiv, integer *jpiv, quadreal *scale)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5, i__6;
    quadreal d__1;
    quadcomplex z__1, z__2, z__3;

    /* Local variables */
    integer i__, j;
    quadreal eps;
    quadcomplex temp;
    extern void  wscal_(integer *, quadcomplex *, 
	    quadcomplex *, integer *), qlabad_(quadreal *, quadreal *);
    extern quadreal qlamch_(char *);
    quadreal bignum;
    extern integer iwamax_(integer *, quadcomplex *, integer *);
    quadreal smlnum;
    extern void  wlaswp_(integer *, quadcomplex *, integer *,
	     integer *, integer *, integer *, integer *);


/*  -- LAPACK auxiliary routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     November 2017 */


/*  ===================================================================== */


/*     Set constant to control overflow */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --rhs;
    --ipiv;
    --jpiv;

    /* Function Body */
    eps = qlamch_("P");
    smlnum = qlamch_("S") / eps;
    bignum = 1. / smlnum;
    qlabad_(&smlnum, &bignum);

/*     Apply permutations IPIV to RHS */

    i__1 = *n - 1;
    wlaswp_(&c__1, &rhs[1], lda, &c__1, &i__1, &ipiv[1], &c__1);

/*     Solve for L part */

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    i__3 = j;
	    i__4 = j;
	    i__5 = j + i__ * a_dim1;
	    i__6 = i__;
	    z__2.r = a[i__5].r * rhs[i__6].r - a[i__5].i * rhs[i__6].i, 
		    z__2.i = a[i__5].r * rhs[i__6].i + a[i__5].i * rhs[i__6]
		    .r;
	    z__1.r = rhs[i__4].r - z__2.r, z__1.i = rhs[i__4].i - z__2.i;
	    rhs[i__3].r = z__1.r, rhs[i__3].i = z__1.i;
/* L10: */
	}
/* L20: */
    }

/*     Solve for U part */

    *scale = 1.;

/*     Check for scaling */

    i__ = iwamax_(n, &rhs[1], &c__1);
    if (smlnum * 2. * z_abs(&rhs[i__]) > z_abs(&a[*n + *n * a_dim1])) {
	d__1 = z_abs(&rhs[i__]);
	z__1.r = .5 / d__1, z__1.i = 0. / d__1;
	temp.r = z__1.r, temp.i = z__1.i;
	wscal_(n, &temp, &rhs[1], &c__1);
	*scale *= temp.r;
    }
    for (i__ = *n; i__ >= 1; --i__) {
	z_div(&z__1, &c_b13, &a[i__ + i__ * a_dim1]);
	temp.r = z__1.r, temp.i = z__1.i;
	i__1 = i__;
	i__2 = i__;
	z__1.r = rhs[i__2].r * temp.r - rhs[i__2].i * temp.i, z__1.i = rhs[
		i__2].r * temp.i + rhs[i__2].i * temp.r;
	rhs[i__1].r = z__1.r, rhs[i__1].i = z__1.i;
	i__1 = *n;
	for (j = i__ + 1; j <= i__1; ++j) {
	    i__2 = i__;
	    i__3 = i__;
	    i__4 = j;
	    i__5 = i__ + j * a_dim1;
	    z__3.r = a[i__5].r * temp.r - a[i__5].i * temp.i, z__3.i = a[i__5]
		    .r * temp.i + a[i__5].i * temp.r;
	    z__2.r = rhs[i__4].r * z__3.r - rhs[i__4].i * z__3.i, z__2.i = 
		    rhs[i__4].r * z__3.i + rhs[i__4].i * z__3.r;
	    z__1.r = rhs[i__3].r - z__2.r, z__1.i = rhs[i__3].i - z__2.i;
	    rhs[i__2].r = z__1.r, rhs[i__2].i = z__1.i;
/* L30: */
	}
/* L40: */
    }

/*     Apply permutations JPIV to the solution (RHS) */

    i__1 = *n - 1;
    wlaswp_(&c__1, &rhs[1], lda, &c__1, &i__1, &jpiv[1], &c_n1);
    return;

/*     End of ZGESC2 */

} /* wgesc2_ */

