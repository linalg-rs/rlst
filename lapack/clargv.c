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

/* > \brief \b CLARGV generates a vector of plane rotations with real cosines and complex sines. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CLARGV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/clargv.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/clargv.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/clargv.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CLARGV( N, X, INCX, Y, INCY, C, INCC ) */

/*       INTEGER            INCC, INCX, INCY, N */
/*       REAL               C( * ) */
/*       COMPLEX            X( * ), Y( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CLARGV generates a vector of complex plane rotations with real */
/* > cosines, determined by elements of the complex vectors x and y. */
/* > For i = 1,2,...,n */
/* > */
/* >    (        c(i)   s(i) ) ( x(i) ) = ( r(i) ) */
/* >    ( -conjg(s(i))  c(i) ) ( y(i) ) = (   0  ) */
/* > */
/* >    where c(i)**2 + ABS(s(i))**2 = 1 */
/* > */
/* > The following conventions are used (these are the same as in CLARTG, */
/* > but differ from the BLAS1 routine CROTG): */
/* >    If y(i)=0, then c(i)=1 and s(i)=0. */
/* >    If x(i)=0, then c(i)=0 and s(i) is chosen so that r(i) is real. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of plane rotations to be generated. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX array, dimension (1+(N-1)*INCX) */
/* >          On entry, the vector x. */
/* >          On exit, x(i) is overwritten by r(i), for i = 1,...,n. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The increment between elements of X. INCX > 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Y */
/* > \verbatim */
/* >          Y is COMPLEX array, dimension (1+(N-1)*INCY) */
/* >          On entry, the vector y. */
/* >          On exit, the sines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] INCY */
/* > \verbatim */
/* >          INCY is INTEGER */
/* >          The increment between elements of Y. INCY > 0. */
/* > \endverbatim */
/* > */
/* > \param[out] C */
/* > \verbatim */
/* >          C is REAL array, dimension (1+(N-1)*INCC) */
/* >          The cosines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] INCC */
/* > \verbatim */
/* >          INCC is INTEGER */
/* >          The increment between elements of C. INCC > 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complexOTHERauxiliary */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  6-6-96 - Modified with a new algorithm by W. Kahan and J. Demmel */
/* > */
/* >  This version has a few statements commented out for thread safety */
/* >  (machine parameters are computed on each entry). 10 feb 03, SJH. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  clargv_(integer *n, complex *x, integer *incx, complex *
	y, integer *incy, real *c__, integer *incc)
{
    /* System generated locals */
    integer i__1, i__2;
    real r__1, r__2, r__3, r__4, r__5, r__6, r__7, r__8, r__9, r__10;
    complex q__1, q__2, q__3;

    /* Local variables */
    real d__;
    complex f, g;
    integer i__, j;
    complex r__;
    real f2, g2;
    integer ic;
    real di;
    complex ff;
    real cs, dr;
    complex fs, gs;
    integer ix, iy;
    complex sn;
    real f2s, g2s, eps, scale;
    integer count;
    real safmn2, safmx2;
    extern real slapy2_(real *, real *), slamch_(char *);
    real safmin;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */

/*     LOGICAL            FIRST */
/*     SAVE               FIRST, SAFMX2, SAFMIN, SAFMN2 */
/*     DATA               FIRST / .TRUE. / */

/*     IF( FIRST ) THEN */
/*        FIRST = .FALSE. */
    /* Parameter adjustments */
    --c__;
    --y;
    --x;

    /* Function Body */
    safmin = slamch_("S");
    eps = slamch_("E");
    r__1 = slamch_("B");
    i__1 = (integer) (M(log)(safmin / eps) / M(log)(slamch_("B")) / 2.f);
    safmn2 = pow_ri(&r__1, &i__1);
    safmx2 = 1.f / safmn2;
/*     END IF */
    ix = 1;
    iy = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	f.r = x[i__2].r, f.i = x[i__2].i;
	i__2 = iy;
	g.r = y[i__2].r, g.i = y[i__2].i;

/*        Use identical algorithm as in CLARTG */

/* Computing MAX */
/* Computing MAX */
	r__7 = (r__1 = f.r, abs(r__1)), r__8 = (r__2 = r_imag(&f), abs(r__2));
/* Computing MAX */
	r__9 = (r__3 = g.r, abs(r__3)), r__10 = (r__4 = r_imag(&g), abs(r__4))
		;
	r__5 = f2cmax(r__7,r__8), r__6 = f2cmax(r__9,r__10);
	scale = f2cmax(r__5,r__6);
	fs.r = f.r, fs.i = f.i;
	gs.r = g.r, gs.i = g.i;
	count = 0;
	if (scale >= safmx2) {
L10:
	    ++count;
	    q__1.r = safmn2 * fs.r, q__1.i = safmn2 * fs.i;
	    fs.r = q__1.r, fs.i = q__1.i;
	    q__1.r = safmn2 * gs.r, q__1.i = safmn2 * gs.i;
	    gs.r = q__1.r, gs.i = q__1.i;
	    scale *= safmn2;
	    if (scale >= safmx2) {
		goto L10;
	    }
	} else if (scale <= safmn2) {
	    if (g.r == 0.f && g.i == 0.f) {
		cs = 1.f;
		sn.r = 0.f, sn.i = 0.f;
		r__.r = f.r, r__.i = f.i;
		goto L50;
	    }
L20:
	    --count;
	    q__1.r = safmx2 * fs.r, q__1.i = safmx2 * fs.i;
	    fs.r = q__1.r, fs.i = q__1.i;
	    q__1.r = safmx2 * gs.r, q__1.i = safmx2 * gs.i;
	    gs.r = q__1.r, gs.i = q__1.i;
	    scale *= safmx2;
	    if (scale <= safmn2) {
		goto L20;
	    }
	}
/* Computing 2nd power */
	r__1 = fs.r;
/* Computing 2nd power */
	r__2 = r_imag(&fs);
	f2 = r__1 * r__1 + r__2 * r__2;
/* Computing 2nd power */
	r__1 = gs.r;
/* Computing 2nd power */
	r__2 = r_imag(&gs);
	g2 = r__1 * r__1 + r__2 * r__2;
	if (f2 <= f2cmax(g2,1.f) * safmin) {

/*           This is a rare case: F is very small. */

	    if (f.r == 0.f && f.i == 0.f) {
		cs = 0.f;
		r__2 = g.r;
		r__3 = r_imag(&g);
		r__1 = slapy2_(&r__2, &r__3);
		r__.r = r__1, r__.i = 0.f;
/*              Do complex/real division explicitly with two real */
/*              divisions */
		r__1 = gs.r;
		r__2 = r_imag(&gs);
		d__ = slapy2_(&r__1, &r__2);
		r__1 = gs.r / d__;
		r__2 = -r_imag(&gs) / d__;
		q__1.r = r__1, q__1.i = r__2;
		sn.r = q__1.r, sn.i = q__1.i;
		goto L50;
	    }
	    r__1 = fs.r;
	    r__2 = r_imag(&fs);
	    f2s = slapy2_(&r__1, &r__2);
/*           G2 and G2S are accurate */
/*           G2 is at least SAFMIN, and G2S is at least SAFMN2 */
	    g2s = M(sqrt)(g2);
/*           Error in CS from underflow in F2S is at most */
/*           UNFL / SAFMN2 .lt. M(sqrt)(UNFL*EPS) .lt. EPS */
/*           If MAX(G2,ONE)=G2, then F2 .lt. G2*SAFMIN, */
/*           and so CS .lt. M(sqrt)(SAFMIN) */
/*           If MAX(G2,ONE)=ONE, then F2 .lt. SAFMIN */
/*           and so CS .lt. M(sqrt)(SAFMIN)/SAFMN2 = M(sqrt)(EPS) */
/*           Therefore, CS = F2S/G2S / M(sqrt)( 1 + (F2S/G2S)**2 ) = F2S/G2S */
	    cs = f2s / g2s;
/*           Make sure abs(FF) = 1 */
/*           Do complex/real division explicitly with 2 real divisions */
/* Computing MAX */
	    r__3 = (r__1 = f.r, abs(r__1)), r__4 = (r__2 = r_imag(&f), abs(
		    r__2));
	    if (f2cmax(r__3,r__4) > 1.f) {
		r__1 = f.r;
		r__2 = r_imag(&f);
		d__ = slapy2_(&r__1, &r__2);
		r__1 = f.r / d__;
		r__2 = r_imag(&f) / d__;
		q__1.r = r__1, q__1.i = r__2;
		ff.r = q__1.r, ff.i = q__1.i;
	    } else {
		dr = safmx2 * f.r;
		di = safmx2 * r_imag(&f);
		d__ = slapy2_(&dr, &di);
		r__1 = dr / d__;
		r__2 = di / d__;
		q__1.r = r__1, q__1.i = r__2;
		ff.r = q__1.r, ff.i = q__1.i;
	    }
	    r__1 = gs.r / g2s;
	    r__2 = -r_imag(&gs) / g2s;
	    q__2.r = r__1, q__2.i = r__2;
	    q__1.r = ff.r * q__2.r - ff.i * q__2.i, q__1.i = ff.r * q__2.i + 
		    ff.i * q__2.r;
	    sn.r = q__1.r, sn.i = q__1.i;
	    q__2.r = cs * f.r, q__2.i = cs * f.i;
	    q__3.r = sn.r * g.r - sn.i * g.i, q__3.i = sn.r * g.i + sn.i * 
		    g.r;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    r__.r = q__1.r, r__.i = q__1.i;
	} else {

/*           This is the most common case. */
/*           Neither F2 nor F2/G2 are less than SAFMIN */
/*           F2S cannot overflow, and it is accurate */

	    f2s = M(sqrt)(g2 / f2 + 1.f);
/*           Do the F2S(real)*FS(complex) multiply with two real */
/*           multiplies */
	    r__1 = f2s * fs.r;
	    r__2 = f2s * r_imag(&fs);
	    q__1.r = r__1, q__1.i = r__2;
	    r__.r = q__1.r, r__.i = q__1.i;
	    cs = 1.f / f2s;
	    d__ = f2 + g2;
/*           Do complex/real division explicitly with two real divisions */
	    r__1 = r__.r / d__;
	    r__2 = r_imag(&r__) / d__;
	    q__1.r = r__1, q__1.i = r__2;
	    sn.r = q__1.r, sn.i = q__1.i;
	    r_cnjg(&q__2, &gs);
	    q__1.r = sn.r * q__2.r - sn.i * q__2.i, q__1.i = sn.r * q__2.i + 
		    sn.i * q__2.r;
	    sn.r = q__1.r, sn.i = q__1.i;
	    if (count != 0) {
		if (count > 0) {
		    i__2 = count;
		    for (j = 1; j <= i__2; ++j) {
			q__1.r = safmx2 * r__.r, q__1.i = safmx2 * r__.i;
			r__.r = q__1.r, r__.i = q__1.i;
/* L30: */
		    }
		} else {
		    i__2 = -count;
		    for (j = 1; j <= i__2; ++j) {
			q__1.r = safmn2 * r__.r, q__1.i = safmn2 * r__.i;
			r__.r = q__1.r, r__.i = q__1.i;
/* L40: */
		    }
		}
	    }
	}
L50:
	c__[ic] = cs;
	i__2 = iy;
	y[i__2].r = sn.r, y[i__2].i = sn.i;
	i__2 = ix;
	x[i__2].r = r__.r, x[i__2].i = r__.i;
	ic += *incc;
	iy += *incy;
	ix += *incx;
/* L60: */
    }
    return;

/*     End of CLARGV */

} /* clargv_ */

