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

/* > \brief \b ZLAR2V applies a vector of plane rotations with doublereal cosines and complex sines from both sides 
to a sequence of 2-by-2 symmetric/Hermitian matrices. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAR2V + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlar2v.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlar2v.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlar2v.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAR2V( N, X, Y, Z, INCX, C, S, INCC ) */

/*       INTEGER            INCC, INCX, N */
/*       DOUBLE PRECISION   C( * ) */
/*       COMPLEX*16         S( * ), X( * ), Y( * ), Z( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLAR2V applies a vector of complex plane rotations with doublereal cosines */
/* > from both sides to a sequence of 2-by-2 complex Hermitian matrices, */
/* > defined by the elements of the vectors x, y and z. For i = 1,2,...,n */
/* > */
/* >    (       x(i)  z(i) ) := */
/* >    ( conjg(z(i)) y(i) ) */
/* > */
/* >      (  c(i) conjg(s(i)) ) (       x(i)  z(i) ) ( c(i) -conjg(s(i)) ) */
/* >      ( -s(i)       c(i)  ) ( conjg(z(i)) y(i) ) ( s(i)        c(i)  ) */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of plane rotations to be applied. */
/* > \endverbatim */
/* > */
/* > \param[in,out] X */
/* > \verbatim */
/* >          X is COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/* >          The vector x; the elements of x are assumed to be doublereal. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Y */
/* > \verbatim */
/* >          Y is COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/* >          The vector y; the elements of y are assumed to be doublereal. */
/* > \endverbatim */
/* > */
/* > \param[in,out] Z */
/* > \verbatim */
/* >          Z is COMPLEX*16 array, dimension (1+(N-1)*INCX) */
/* >          The vector z. */
/* > \endverbatim */
/* > */
/* > \param[in] INCX */
/* > \verbatim */
/* >          INCX is INTEGER */
/* >          The increment between elements of X, Y and Z. INCX > 0. */
/* > \endverbatim */
/* > */
/* > \param[in] C */
/* > \verbatim */
/* >          C is DOUBLE PRECISION array, dimension (1+(N-1)*INCC) */
/* >          The cosines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] S */
/* > \verbatim */
/* >          S is COMPLEX*16 array, dimension (1+(N-1)*INCC) */
/* >          The sines of the plane rotations. */
/* > \endverbatim */
/* > */
/* > \param[in] INCC */
/* > \verbatim */
/* >          INCC is INTEGER */
/* >          The increment between elements of C and S. INCC > 0. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup complex16OTHERauxiliary */

/*  ===================================================================== */
void  klar2v_(integer *n, halfcomplex *x, halfcomplex *y, 
	halfcomplex *z__, integer *incx, halfreal *c__, halfcomplex *s, 
	integer *incc)
{
    /* System generated locals */
    integer i__1, i__2;
    halfreal d__1;
    halfcomplex z__1, z__2, z__3, z__4, z__5;

    /* Local variables */
    integer i__;
    halfcomplex t2, t3, t4;
    halfreal t5, t6;
    integer ic;
    halfreal ci;
    halfcomplex si;
    integer ix;
    halfreal xi, yi;
    halfcomplex zi;
    halfreal t1i, t1r, sii, zii, sir, zir;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --s;
    --c__;
    --z__;
    --y;
    --x;

    /* Function Body */
    ix = 1;
    ic = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	i__2 = ix;
	xi = x[i__2].r;
	i__2 = ix;
	yi = y[i__2].r;
	i__2 = ix;
	zi.r = z__[i__2].r, zi.i = z__[i__2].i;
	zir = zi.r;
	zii = d_imag(&zi);
	ci = c__[ic];
	i__2 = ic;
	si.r = s[i__2].r, si.i = s[i__2].i;
	sir = si.r;
	sii = d_imag(&si);
	t1r = sir * zir - sii * zii;
	t1i = sir * zii + sii * zir;
	z__1.r = ci * zi.r, z__1.i = ci * zi.i;
	t2.r = z__1.r, t2.i = z__1.i;
	d_cnjg(&z__3, &si);
	z__2.r = xi * z__3.r, z__2.i = xi * z__3.i;
	z__1.r = t2.r - z__2.r, z__1.i = t2.i - z__2.i;
	t3.r = z__1.r, t3.i = z__1.i;
	d_cnjg(&z__2, &t2);
	z__3.r = yi * si.r, z__3.i = yi * si.i;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	t4.r = z__1.r, t4.i = z__1.i;
	t5 = ci * xi + t1r;
	t6 = ci * yi - t1r;
	i__2 = ix;
	d__1 = ci * t5 + (sir * t4.r + sii * d_imag(&t4));
	x[i__2].r = d__1, x[i__2].i = 0.;
	i__2 = ix;
	d__1 = ci * t6 - (sir * t3.r - sii * d_imag(&t3));
	y[i__2].r = d__1, y[i__2].i = 0.;
	i__2 = ix;
	z__2.r = ci * t3.r, z__2.i = ci * t3.i;
	d_cnjg(&z__4, &si);
	z__5.r = t6, z__5.i = t1i;
	z__3.r = z__4.r * z__5.r - z__4.i * z__5.i, z__3.i = z__4.r * z__5.i 
		+ z__4.i * z__5.r;
	z__1.r = z__2.r + z__3.r, z__1.i = z__2.i + z__3.i;
	z__[i__2].r = z__1.r, z__[i__2].i = z__1.i;
	ix += *incx;
	ic += *incc;
/* L10: */
    }
    return;

/*     End of ZLAR2V */

} /* klar2v_ */

