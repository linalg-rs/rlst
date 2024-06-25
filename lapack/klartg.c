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

/* > \brief \b ZLARTG generates a plane rotation with doublereal cosine and complex sine. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLARTG + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlartg.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlartg.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlartg.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARTG( F, G, CS, SN, R ) */

/*       DOUBLE PRECISION   CS */
/*       COMPLEX*16         F, G, R, SN */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARTG generates a plane rotation so that */
/* > */
/* >    [  CS  SN  ]     [ F ]     [ R ] */
/* >    [  __      ]  .  [   ]  =  [   ]   where CS**2 + |SN|**2 = 1. */
/* >    [ -SN  CS  ]     [ G ]     [ 0 ] */
/* > */
/* > This is a faster version of the BLAS1 routine ZROTG, except for */
/* > the following differences: */
/* >    F and G are unchanged on return. */
/* >    If G=0, then CS=1 and SN=0. */
/* >    If F=0, then CS=0 and SN is chosen so that R is doublereal. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] F */
/* > \verbatim */
/* >          F is COMPLEX*16 */
/* >          The first component of vector to be rotated. */
/* > \endverbatim */
/* > */
/* > \param[in] G */
/* > \verbatim */
/* >          G is COMPLEX*16 */
/* >          The second component of vector to be rotated. */
/* > \endverbatim */
/* > */
/* > \param[out] CS */
/* > \verbatim */
/* >          CS is DOUBLE PRECISION */
/* >          The cosine of the rotation. */
/* > \endverbatim */
/* > */
/* > \param[out] SN */
/* > \verbatim */
/* >          SN is COMPLEX*16 */
/* >          The sine of the rotation. */
/* > \endverbatim */
/* > */
/* > \param[out] R */
/* > \verbatim */
/* >          R is COMPLEX*16 */
/* >          The nonzero component of the rotated vector. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  3-5-96 - Modified with a new algorithm by W. Kahan and J. Demmel */
/* > */
/* >  This version has a few statements commented out for thread safety */
/* >  (machine parameters are computed on each entry). 10 feb 03, SJH. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  klartg_(halfcomplex *f, halfcomplex *g, halfreal *
	cs, halfcomplex *sn, halfcomplex *r__)
{
    /* System generated locals */
    integer i__1;
    halfreal d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8, d__9, d__10;
    halfcomplex z__1, z__2, z__3;

    /* Local variables */
    halfreal d__;
    integer i__;
    halfreal f2, g2;
    halfcomplex ff;
    halfreal di, dr;
    halfcomplex fs, gs;
    halfreal f2s, g2s, eps, scale;
    integer count;
    halfreal safmn2;
    extern halfreal hlapy2_(halfreal *, halfreal *);
    halfreal safmx2;
    extern halfreal hlamch_(char *);
    extern logical hisnan_(halfreal *);
    halfreal safmin;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */

/*     LOGICAL            FIRST */

    safmin = hlamch_("S");
    eps = hlamch_("E");
    d__1 = hlamch_("B");
    i__1 = (integer) (M(log)(safmin / eps) / M(log)(hlamch_("B")) / 2.);
    safmn2 = pow_di(&d__1, &i__1);
    safmx2 = 1. / safmn2;
/* Computing MAX */
/* Computing MAX */
    d__7 = (d__1 = f->r, abs(d__1)), d__8 = (d__2 = d_imag(f), abs(d__2));
/* Computing MAX */
    d__9 = (d__3 = g->r, abs(d__3)), d__10 = (d__4 = d_imag(g), abs(d__4));
    d__5 = f2cmax(d__7,d__8), d__6 = f2cmax(d__9,d__10);
    scale = f2cmax(d__5,d__6);
    fs.r = f->r, fs.i = f->i;
    gs.r = g->r, gs.i = g->i;
    count = 0;
    if (scale >= safmx2) {
L10:
	++count;
	z__1.r = safmn2 * fs.r, z__1.i = safmn2 * fs.i;
	fs.r = z__1.r, fs.i = z__1.i;
	z__1.r = safmn2 * gs.r, z__1.i = safmn2 * gs.i;
	gs.r = z__1.r, gs.i = z__1.i;
	scale *= safmn2;
	if (scale >= safmx2) {
	    goto L10;
	}
    } else if (scale <= safmn2) {
	d__1 = z_abs(g);
	if (g->r == 0. && g->i == 0. || hisnan_(&d__1)) {
	    *cs = 1.;
	    sn->r = 0., sn->i = 0.;
	    r__->r = f->r, r__->i = f->i;
	    return;
	}
L20:
	--count;
	z__1.r = safmx2 * fs.r, z__1.i = safmx2 * fs.i;
	fs.r = z__1.r, fs.i = z__1.i;
	z__1.r = safmx2 * gs.r, z__1.i = safmx2 * gs.i;
	gs.r = z__1.r, gs.i = z__1.i;
	scale *= safmx2;
	if (scale <= safmn2) {
	    goto L20;
	}
    }
/* Computing 2nd power */
    d__1 = fs.r;
/* Computing 2nd power */
    d__2 = d_imag(&fs);
    f2 = d__1 * d__1 + d__2 * d__2;
/* Computing 2nd power */
    d__1 = gs.r;
/* Computing 2nd power */
    d__2 = d_imag(&gs);
    g2 = d__1 * d__1 + d__2 * d__2;
    if (f2 <= f2cmax(g2,1.) * safmin) {

/*        This is a rare case: F is very small. */

	if (f->r == 0. && f->i == 0.) {
	    *cs = 0.;
	    d__2 = g->r;
	    d__3 = d_imag(g);
	    d__1 = hlapy2_(&d__2, &d__3);
	    r__->r = d__1, r__->i = 0.;
/*           Do complex/doublereal division explicitly with two doublereal divisions */
	    d__1 = gs.r;
	    d__2 = d_imag(&gs);
	    d__ = hlapy2_(&d__1, &d__2);
	    d__1 = gs.r / d__;
	    d__2 = -d_imag(&gs) / d__;
	    z__1.r = d__1, z__1.i = d__2;
	    sn->r = z__1.r, sn->i = z__1.i;
	    return;
	}
	d__1 = fs.r;
	d__2 = d_imag(&fs);
	f2s = hlapy2_(&d__1, &d__2);
/*        G2 and G2S are accurate */
/*        G2 is at least SAFMIN, and G2S is at least SAFMN2 */
	g2s = M(sqrt)(g2);
/*        Error in CS from underflow in F2S is at most */
/*        UNFL / SAFMN2 .lt. M(sqrt)(UNFL*EPS) .lt. EPS */
/*        If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN, */
/*        and so CS .lt. M(sqrt)(SAFMIN) */
/*        If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN */
/*        and so CS .lt. M(sqrt)(SAFMIN)/SAFMN2 = M(sqrt)(EPS) */
/*        Therefore, CS = F2S/G2S / M(sqrt)( 1 + (F2S/G2S)**2 ) = F2S/G2S */
	*cs = f2s / g2s;
/*        Make sure abs(FF) = 1 */
/*        Do complex/doublereal division explicitly with 2 doublereal divisions */
/* Computing MAX */
	d__3 = (d__1 = f->r, abs(d__1)), d__4 = (d__2 = d_imag(f), abs(d__2));
	if (f2cmax(d__3,d__4) > 1.) {
	    d__1 = f->r;
	    d__2 = d_imag(f);
	    d__ = hlapy2_(&d__1, &d__2);
	    d__1 = f->r / d__;
	    d__2 = d_imag(f) / d__;
	    z__1.r = d__1, z__1.i = d__2;
	    ff.r = z__1.r, ff.i = z__1.i;
	} else {
	    dr = safmx2 * f->r;
	    di = safmx2 * d_imag(f);
	    d__ = hlapy2_(&dr, &di);
	    d__1 = dr / d__;
	    d__2 = di / d__;
	    z__1.r = d__1, z__1.i = d__2;
	    ff.r = z__1.r, ff.i = z__1.i;
	}
	d__1 = gs.r / g2s;
	d__2 = -d_imag(&gs) / g2s;
	z__2.r = d__1, z__2.i = d__2;
	z__1.r = ff.r * z__2.r - ff.i * z__2.i, z__1.i = ff.r * z__2.i + ff.i 
		* z__2.r;
	sn->r = z__1.r, sn->i = z__1.i;
	z__2.r = *cs * f->r, z__2.i = *cs * f->i;
	z__3.r = sn->r * g->r - sn->i * g->i, z__3.i = sn->r * g->i + sn->i * 
		g->r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	r__->r = z__1.r, r__->i = z__1.i;
    } else {

/*        This is the most common case. */
/*        Neither F2 nor F2/G2 are less than SAFMIN */
/*        F2S cannot overflow, and it is accurate */

	f2s = M(sqrt)(g2 / f2 + 1.);
/*        Do the F2S(doublereal)*FS(complex) multiply with two doublereal multiplies */
	d__1 = f2s * fs.r;
	d__2 = f2s * d_imag(&fs);
	z__1.r = d__1, z__1.i = d__2;
	r__->r = z__1.r, r__->i = z__1.i;
	*cs = 1. / f2s;
	d__ = f2 + g2;
/*        Do complex/doublereal division explicitly with two doublereal divisions */
	d__1 = r__->r / d__;
	d__2 = d_imag(r__) / d__;
	z__1.r = d__1, z__1.i = d__2;
	sn->r = z__1.r, sn->i = z__1.i;
	d_cnjg(&z__2, &gs);
	z__1.r = sn->r * z__2.r - sn->i * z__2.i, z__1.i = sn->r * z__2.i + 
		sn->i * z__2.r;
	sn->r = z__1.r, sn->i = z__1.i;
	if (count != 0) {
	    if (count > 0) {
		i__1 = count;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    z__1.r = safmx2 * r__->r, z__1.i = safmx2 * r__->i;
		    r__->r = z__1.r, r__->i = z__1.i;
/* L30: */
		}
	    } else {
		i__1 = -count;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    z__1.r = safmn2 * r__->r, z__1.i = safmn2 * r__->i;
		    r__->r = z__1.r, r__->i = z__1.i;
/* L40: */
		}
	    }
	}
    }
    return;

/*     End of ZLARTG */

} /* klartg_ */

