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

#define __LAPACK_PRECISION_HALF
#include "f2c.h"

/* Table of constant values */

static integer c__1 = 1;
static halfcomplex c_b10 = {-1.,-0.};

/* > \brief \b ZGETC2 computes the LU factorization with complete pivoting of the general n-by-n matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZGETC2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zgetc2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zgetc2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zgetc2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZGETC2( N, A, LDA, IPIV, JPIV, INFO ) */

/*       INTEGER            INFO, LDA, N */
/*       INTEGER            IPIV( * ), JPIV( * ) */
/*       COMPLEX*16         A( LDA, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZGETC2 computes an LU factorization, using complete pivoting, of the */
/* > n-by-n matrix A. The factorization has the form A = P * L * U * Q, */
/* > where P and Q are permutation matrices, L is lower triangular with */
/* > unit diagonal elements and U is upper triangular. */
/* > */
/* > This is a level 1 BLAS version of the algorithm. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The order of the matrix A. N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX*16 array, dimension (LDA, N) */
/* >          On entry, the n-by-n matrix to be factored. */
/* >          On exit, the factors L and U from the factorization */
/* >          A = P*L*U*Q; the unit diagonal elements of L are not stored. */
/* >          If U(k, k) appears to be less than SMIN, U(k, k) is given the */
/* >          value of SMIN, giving a nonsingular perturbed system. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1, N). */
/* > \endverbatim */
/* > */
/* > \param[out] IPIV */
/* > \verbatim */
/* >          IPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= i <= N, row i of the */
/* >          matrix has been interchanged with row IPIV(i). */
/* > \endverbatim */
/* > */
/* > \param[out] JPIV */
/* > \verbatim */
/* >          JPIV is INTEGER array, dimension (N). */
/* >          The pivot indices; for 1 <= j <= N, column j of the */
/* >          matrix has been interchanged with column JPIV(j). */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >           = 0: successful exit */
/* >           > 0: if INFO = k, U(k, k) is likely to produce overflow if */
/* >                one tries to solve for x in Ax = b. So U is perturbed */
/* >                to avoid the overflow. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup complex16GEauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Bo Kagstrom and Peter Poromaa, Department of Computing Science, */
/* >     Umea University, S-901 87 Umea, Sweden. */

/*  ===================================================================== */
void  kgetc2_(integer *n, halfcomplex *a, integer *lda, 
	integer *ipiv, integer *jpiv, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    halfreal d__1;
    halfcomplex z__1;

    /* Local variables */
    integer i__, j, ip, jp;
    halfreal eps;
    integer ipv, jpv;
    halfreal smin, xmax;
    extern void  kgeru_(integer *, integer *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), kswap_(integer *, halfcomplex *, 
	    integer *, halfcomplex *, integer *), hlabad_(halfreal *, 
	    halfreal *);
    extern halfreal hlamch_(char *);
    halfreal bignum, smlnum;


/*  -- LAPACK auxiliary routine (version 3.8.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    --jpiv;

    /* Function Body */
    *info = 0;

/*     Quick return if possible */

    if (*n == 0) {
	return;
    }

/*     Set constants to control overflow */

    eps = hlamch_("P");
    smlnum = hlamch_("S") / eps;
    bignum = 1. / smlnum;
    hlabad_(&smlnum, &bignum);

/*     Handle the case N=1 by itself */

    if (*n == 1) {
	ipiv[1] = 1;
	jpiv[1] = 1;
	if (z_abs(&a[a_dim1 + 1]) < smlnum) {
	    *info = 1;
	    i__1 = a_dim1 + 1;
	    z__1.r = smlnum, z__1.i = 0.;
	    a[i__1].r = z__1.r, a[i__1].i = z__1.i;
	}
	return;
    }

/*     Factorize A using complete pivoting. */
/*     Set pivots less than SMIN to SMIN */

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Find f2cmax element in matrix A */

	xmax = 0.;
	i__2 = *n;
	for (ip = i__; ip <= i__2; ++ip) {
	    i__3 = *n;
	    for (jp = i__; jp <= i__3; ++jp) {
		if (z_abs(&a[ip + jp * a_dim1]) >= xmax) {
		    xmax = z_abs(&a[ip + jp * a_dim1]);
		    ipv = ip;
		    jpv = jp;
		}
/* L10: */
	    }
/* L20: */
	}
	if (i__ == 1) {
/* Computing MAX */
	    d__1 = eps * xmax;
	    smin = f2cmax(d__1,smlnum);
	}

/*        Swap rows */

	if (ipv != i__) {
	    kswap_(n, &a[ipv + a_dim1], lda, &a[i__ + a_dim1], lda);
	}
	ipiv[i__] = ipv;

/*        Swap columns */

	if (jpv != i__) {
	    kswap_(n, &a[jpv * a_dim1 + 1], &c__1, &a[i__ * a_dim1 + 1], &
		    c__1);
	}
	jpiv[i__] = jpv;

/*        Check for singularity */

	if (z_abs(&a[i__ + i__ * a_dim1]) < smin) {
	    *info = i__;
	    i__2 = i__ + i__ * a_dim1;
	    z__1.r = smin, z__1.i = 0.;
	    a[i__2].r = z__1.r, a[i__2].i = z__1.i;
	}
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    i__3 = j + i__ * a_dim1;
	    z_div(&z__1, &a[j + i__ * a_dim1], &a[i__ + i__ * a_dim1]);
	    a[i__3].r = z__1.r, a[i__3].i = z__1.i;
/* L30: */
	}
	i__2 = *n - i__;
	i__3 = *n - i__;
	kgeru_(&i__2, &i__3, &c_b10, &a[i__ + 1 + i__ * a_dim1], &c__1, &a[
		i__ + (i__ + 1) * a_dim1], lda, &a[i__ + 1 + (i__ + 1) * 
		a_dim1], lda);
/* L40: */
    }

    if (z_abs(&a[*n + *n * a_dim1]) < smin) {
	*info = *n;
	i__1 = *n + *n * a_dim1;
	z__1.r = smin, z__1.i = 0.;
	a[i__1].r = z__1.r, a[i__1].i = z__1.i;
    }

/*     Set last pivots to N */

    ipiv[*n] = *n;
    jpiv[*n] = *n;

    return;

/*     End of ZGETC2 */

} /* kgetc2_ */

