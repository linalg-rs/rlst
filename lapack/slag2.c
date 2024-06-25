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

/* > \brief \b SLAG2 computes the eigenvalues of a 2-by-2 generalized eigenvalue problem, with scaling as nece
ssary to avoid over-/underflow. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SLAG2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slag2.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slag2.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slag2.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE SLAG2( A, LDA, B, LDB, SAFMIN, SCALE1, SCALE2, WR1, */
/*                         WR2, WI ) */

/*       INTEGER            LDA, LDB */
/*       REAL               SAFMIN, SCALE1, SCALE2, WI, WR1, WR2 */
/*       REAL               A( LDA, * ), B( LDB, * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SLAG2 computes the eigenvalues of a 2 x 2 generalized eigenvalue */
/* > problem  A - w B, with scaling as necessary to avoid over-/underflow. */
/* > */
/* > The scaling factor "s" results in a modified eigenvalue equation */
/* > */
/* >     s A - w B */
/* > */
/* > where  s  is a non-negative scaling factor chosen so that  w,  w B, */
/* > and  s A  do not overflow and, if possible, do not underflow, either. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] A */
/* > \verbatim */
/* >          A is REAL array, dimension (LDA, 2) */
/* >          On entry, the 2 x 2 matrix A.  It is assumed that its 1-norm */
/* >          is less than 1/SAFMIN.  Entries less than */
/* >          M(sqrt)(SAFMIN)*norm(A) are subject to being treated as zero. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= 2. */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is REAL array, dimension (LDB, 2) */
/* >          On entry, the 2 x 2 upper triangular matrix B.  It is */
/* >          assumed that the one-norm of B is less than 1/SAFMIN.  The */
/* >          diagonals should be at least M(sqrt)(SAFMIN) times the largest */
/* >          element of B (in absolute value); if a diagonal is smaller */
/* >          than that, then  +/- M(sqrt)(SAFMIN) will be used instead of */
/* >          that diagonal. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= 2. */
/* > \endverbatim */
/* > */
/* > \param[in] SAFMIN */
/* > \verbatim */
/* >          SAFMIN is REAL */
/* >          The smallest positive number s.t. 1/SAFMIN does not */
/* >          overflow.  (This should always be SLAMCH('S') -- it is an */
/* >          argument in order to avoid having to call SLAMCH frequently.) */
/* > \endverbatim */
/* > */
/* > \param[out] SCALE1 */
/* > \verbatim */
/* >          SCALE1 is REAL */
/* >          A scaling factor used to avoid over-/underflow in the */
/* >          eigenvalue equation which defines the first eigenvalue.  If */
/* >          the eigenvalues are complex, then the eigenvalues are */
/* >          ( WR1  +/-  WI i ) / SCALE1  (which may lie outside the */
/* >          exponent range of the machine), SCALE1=SCALE2, and SCALE1 */
/* >          will always be positive.  If the eigenvalues are real, then */
/* >          the first (real) eigenvalue is  WR1 / SCALE1 , but this may */
/* >          overflow or underflow, and in fact, SCALE1 may be zero or */
/* >          less than the underflow threshold if the exact eigenvalue */
/* >          is sufficiently large. */
/* > \endverbatim */
/* > */
/* > \param[out] SCALE2 */
/* > \verbatim */
/* >          SCALE2 is REAL */
/* >          A scaling factor used to avoid over-/underflow in the */
/* >          eigenvalue equation which defines the second eigenvalue.  If */
/* >          the eigenvalues are complex, then SCALE2=SCALE1.  If the */
/* >          eigenvalues are real, then the second (real) eigenvalue is */
/* >          WR2 / SCALE2 , but this may overflow or underflow, and in */
/* >          fact, SCALE2 may be zero or less than the underflow */
/* >          threshold if the exact eigenvalue is sufficiently large. */
/* > \endverbatim */
/* > */
/* > \param[out] WR1 */
/* > \verbatim */
/* >          WR1 is REAL */
/* >          If the eigenvalue is real, then WR1 is SCALE1 times the */
/* >          eigenvalue closest to the (2,2) element of A B**(-1).  If the */
/* >          eigenvalue is complex, then WR1=WR2 is SCALE1 times the real */
/* >          part of the eigenvalues. */
/* > \endverbatim */
/* > */
/* > \param[out] WR2 */
/* > \verbatim */
/* >          WR2 is REAL */
/* >          If the eigenvalue is real, then WR2 is SCALE2 times the */
/* >          other eigenvalue.  If the eigenvalue is complex, then */
/* >          WR1=WR2 is SCALE1 times the real part of the eigenvalues. */
/* > \endverbatim */
/* > */
/* > \param[out] WI */
/* > \verbatim */
/* >          WI is REAL */
/* >          If the eigenvalue is real, then WI is zero.  If the */
/* >          eigenvalue is complex, then WI is SCALE1 times the imaginary */
/* >          part of the eigenvalues.  WI will always be non-negative. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2016 */

/* > \ingroup realOTHERauxiliary */

/*  ===================================================================== */
void  slag2_(real *a, integer *lda, real *b, integer *ldb, 
	real *safmin, real *scale1, real *scale2, real *wr1, real *wr2, real *
	wi)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset;
    real r__1, r__2, r__3, r__4, r__5, r__6;

    /* Local variables */
    real r__, c1, c2, c3, c4, c5, s1, s2, a11, a12, a21, a22, b11, b12, b22, 
	    pp, qq, ss, as11, as12, as22, sum, abi22, diff, bmin, wbig, wabs, 
	    wdet, binv11, binv22, discr, anorm, bnorm, bsize, shift, rtmin, 
	    rtmax, wsize, ascale, bscale, wscale, safmax, wsmall;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    rtmin = M(sqrt)(*safmin);
    rtmax = 1.f / rtmin;
    safmax = 1.f / *safmin;

/*     Scale A */

/* Computing MAX */
    r__5 = (r__1 = a[a_dim1 + 1], abs(r__1)) + (r__2 = a[a_dim1 + 2], abs(
	    r__2)), r__6 = (r__3 = a[(a_dim1 << 1) + 1], abs(r__3)) + (r__4 = 
	    a[(a_dim1 << 1) + 2], abs(r__4)), r__5 = f2cmax(r__5,r__6);
    anorm = f2cmax(r__5,*safmin);
    ascale = 1.f / anorm;
    a11 = ascale * a[a_dim1 + 1];
    a21 = ascale * a[a_dim1 + 2];
    a12 = ascale * a[(a_dim1 << 1) + 1];
    a22 = ascale * a[(a_dim1 << 1) + 2];

/*     Perturb B if necessary to insure non-singularity */

    b11 = b[b_dim1 + 1];
    b12 = b[(b_dim1 << 1) + 1];
    b22 = b[(b_dim1 << 1) + 2];
/* Computing MAX */
    r__1 = abs(b11), r__2 = abs(b12), r__1 = f2cmax(r__1,r__2), r__2 = abs(b22), 
	    r__1 = f2cmax(r__1,r__2);
    bmin = rtmin * f2cmax(r__1,rtmin);
    if (abs(b11) < bmin) {
	b11 = r_sign(&bmin, &b11);
    }
    if (abs(b22) < bmin) {
	b22 = r_sign(&bmin, &b22);
    }

/*     Scale B */

/* Computing MAX */
    r__1 = abs(b11), r__2 = abs(b12) + abs(b22), r__1 = f2cmax(r__1,r__2);
    bnorm = f2cmax(r__1,*safmin);
/* Computing MAX */
    r__1 = abs(b11), r__2 = abs(b22);
    bsize = f2cmax(r__1,r__2);
    bscale = 1.f / bsize;
    b11 *= bscale;
    b12 *= bscale;
    b22 *= bscale;

/*     Compute larger eigenvalue by method described by C. van Loan */

/*     ( AS is A shifted by -SHIFT*B ) */

    binv11 = 1.f / b11;
    binv22 = 1.f / b22;
    s1 = a11 * binv11;
    s2 = a22 * binv22;
    if (abs(s1) <= abs(s2)) {
	as12 = a12 - s1 * b12;
	as22 = a22 - s1 * b22;
	ss = a21 * (binv11 * binv22);
	abi22 = as22 * binv22 - ss * b12;
	pp = abi22 * .5f;
	shift = s1;
    } else {
	as12 = a12 - s2 * b12;
	as11 = a11 - s2 * b11;
	ss = a21 * (binv11 * binv22);
	abi22 = -ss * b12;
	pp = (as11 * binv11 + abi22) * .5f;
	shift = s2;
    }
    qq = ss * as12;
    if ((r__1 = pp * rtmin, abs(r__1)) >= 1.f) {
/* Computing 2nd power */
	r__1 = rtmin * pp;
	discr = r__1 * r__1 + qq * *safmin;
	r__ = M(sqrt)((abs(discr))) * rtmax;
    } else {
/* Computing 2nd power */
	r__1 = pp;
	if (r__1 * r__1 + abs(qq) <= *safmin) {
/* Computing 2nd power */
	    r__1 = rtmax * pp;
	    discr = r__1 * r__1 + qq * safmax;
	    r__ = M(sqrt)((abs(discr))) * rtmin;
	} else {
/* Computing 2nd power */
	    r__1 = pp;
	    discr = r__1 * r__1 + qq;
	    r__ = M(sqrt)((abs(discr)));
	}
    }

/*     Note: the test of R in the following IF is to cover the case when */
/*           DISCR is small and negative and is flushed to zero during */
/*           the calculation of R.  On machines which have a consistent */
/*           flush-to-zero threshold and handle numbers above that */
/*           threshold correctly, it would not be necessary. */

    if (discr >= 0.f || r__ == 0.f) {
	sum = pp + r_sign(&r__, &pp);
	diff = pp - r_sign(&r__, &pp);
	wbig = shift + sum;

/*        Compute smaller eigenvalue */

	wsmall = shift + diff;
/* Computing MAX */
	r__1 = abs(wsmall);
	if (abs(wbig) * .5f > f2cmax(r__1,*safmin)) {
	    wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
	    wsmall = wdet / wbig;
	}

/*        Choose (real) eigenvalue closest to 2,2 element of A*B**(-1) */
/*        for WR1. */

	if (pp > abi22) {
	    *wr1 = f2cmin(wbig,wsmall);
	    *wr2 = f2cmax(wbig,wsmall);
	} else {
	    *wr1 = f2cmax(wbig,wsmall);
	    *wr2 = f2cmin(wbig,wsmall);
	}
	*wi = 0.f;
    } else {

/*        Complex eigenvalues */

	*wr1 = shift + pp;
	*wr2 = *wr1;
	*wi = r__;
    }

/*     Further scaling to avoid underflow and overflow in computing */
/*     SCALE1 and overflow in computing w*B. */

/*     This scale factor (WSCALE) is bounded from above using C1 and C2, */
/*     and from below using C3 and C4. */
/*        C1 implements the condition  s A  must never overflow. */
/*        C2 implements the condition  w B  must never overflow. */
/*        C3, with C2, */
/*           implement the condition that s A - w B must never overflow. */
/*        C4 implements the condition  s    should not underflow. */
/*        C5 implements the condition  f2cmax(s,|w|) should be at least 2. */

    c1 = bsize * (*safmin * f2cmax(1.f,ascale));
    c2 = *safmin * f2cmax(1.f,bnorm);
    c3 = bsize * *safmin;
    if (ascale <= 1.f && bsize <= 1.f) {
/* Computing MIN */
	r__1 = 1.f, r__2 = ascale / *safmin * bsize;
	c4 = f2cmin(r__1,r__2);
    } else {
	c4 = 1.f;
    }
    if (ascale <= 1.f || bsize <= 1.f) {
/* Computing MIN */
	r__1 = 1.f, r__2 = ascale * bsize;
	c5 = f2cmin(r__1,r__2);
    } else {
	c5 = 1.f;
    }

/*     Scale first eigenvalue */

    wabs = abs(*wr1) + abs(*wi);
/* Computing MAX */
/* Computing MIN */
    r__3 = c4, r__4 = f2cmax(wabs,c5) * .5f;
    r__1 = f2cmax(*safmin,c1), r__2 = (wabs * c2 + c3) * 1.0000100000000001f, 
	    r__1 = f2cmax(r__1,r__2), r__2 = f2cmin(r__3,r__4);
    wsize = f2cmax(r__1,r__2);
    if (wsize != 1.f) {
	wscale = 1.f / wsize;
	if (wsize > 1.f) {
	    *scale1 = f2cmax(ascale,bsize) * wscale * f2cmin(ascale,bsize);
	} else {
	    *scale1 = f2cmin(ascale,bsize) * wscale * f2cmax(ascale,bsize);
	}
	*wr1 *= wscale;
	if (*wi != 0.f) {
	    *wi *= wscale;
	    *wr2 = *wr1;
	    *scale2 = *scale1;
	}
    } else {
	*scale1 = ascale * bsize;
	*scale2 = *scale1;
    }

/*     Scale second eigenvalue (if real) */

    if (*wi == 0.f) {
/* Computing MAX */
/* Computing MIN */
/* Computing MAX */
	r__5 = abs(*wr2);
	r__3 = c4, r__4 = f2cmax(r__5,c5) * .5f;
	r__1 = f2cmax(*safmin,c1), r__2 = (abs(*wr2) * c2 + c3) * 
		1.0000100000000001f, r__1 = f2cmax(r__1,r__2), r__2 = f2cmin(r__3,
		r__4);
	wsize = f2cmax(r__1,r__2);
	if (wsize != 1.f) {
	    wscale = 1.f / wsize;
	    if (wsize > 1.f) {
		*scale2 = f2cmax(ascale,bsize) * wscale * f2cmin(ascale,bsize);
	    } else {
		*scale2 = f2cmin(ascale,bsize) * wscale * f2cmax(ascale,bsize);
	    }
	    *wr2 *= wscale;
	} else {
	    *scale2 = ascale * bsize;
	}
    }

/*     End of SLAG2 */

    return;
} /* slag2_ */

