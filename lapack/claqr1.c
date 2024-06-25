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

/* > \brief \b CLAQR1 sets a scalar multiple of the first column of the product of 2-by-2 or 3-by-3 matrix H a
nd specified shifts. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download CLAQR1 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/claqr1.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/claqr1.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/claqr1.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CLAQR1( N, H, LDH, S1, S2, V ) */

/*       COMPLEX            S1, S2 */
/*       INTEGER            LDH, N */
/*       COMPLEX            H( LDH, * ), V( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* >      Given a 2-by-2 or 3-by-3 matrix H, CLAQR1 sets v to a */
/* >      scalar multiple of the first column of the product */
/* > */
/* >      (*)  K = (H - s1*I)*(H - s2*I) */
/* > */
/* >      scaling to avoid overflows and most underflows. */
/* > */
/* >      This is useful for starting double implicit shift bulges */
/* >      in the QR algorithm. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >              Order of the matrix H. N must be either 2 or 3. */
/* > \endverbatim */
/* > */
/* > \param[in] H */
/* > \verbatim */
/* >          H is COMPLEX array, dimension (LDH,N) */
/* >              The 2-by-2 or 3-by-3 matrix H in (*). */
/* > \endverbatim */
/* > */
/* > \param[in] LDH */
/* > \verbatim */
/* >          LDH is INTEGER */
/* >              The leading dimension of H as declared in */
/* >              the calling procedure.  LDH.GE.N */
/* > \endverbatim */
/* > */
/* > \param[in] S1 */
/* > \verbatim */
/* >          S1 is COMPLEX */
/* > \endverbatim */
/* > */
/* > \param[in] S2 */
/* > \verbatim */
/* >          S2 is COMPLEX */
/* > */
/* >          S1 and S2 are the shifts defining K in (*) above. */
/* > \endverbatim */
/* > */
/* > \param[out] V */
/* > \verbatim */
/* >          V is COMPLEX array, dimension (N) */
/* >              A scalar multiple of the first column of the */
/* >              matrix K in (*). */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2017 */

/* > \ingroup complexOTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >       Karen Braman and Ralph Byers, Department of Mathematics, */
/* >       University of Kansas, USA */
/* > */
/*  ===================================================================== */
void  claqr1_(integer *n, complex *h__, integer *ldh, complex *
	s1, complex *s2, complex *v)
{
    /* System generated locals */
    integer h_dim1, h_offset, i__1, i__2, i__3, i__4;
    real r__1, r__2, r__3, r__4, r__5, r__6;
    complex q__1, q__2, q__3, q__4, q__5, q__6, q__7, q__8;

    /* Local variables */
    real s;
    complex h21s, h31s;


/*  -- LAPACK auxiliary routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/*  ================================================================ */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --v;

    /* Function Body */
    if (*n == 2) {
	i__1 = h_dim1 + 1;
	q__2.r = h__[i__1].r - s2->r, q__2.i = h__[i__1].i - s2->i;
	q__1.r = q__2.r, q__1.i = q__2.i;
	i__2 = h_dim1 + 2;
	s = (r__1 = q__1.r, abs(r__1)) + (r__2 = r_imag(&q__1), abs(r__2)) + (
		(r__3 = h__[i__2].r, abs(r__3)) + (r__4 = r_imag(&h__[h_dim1 
		+ 2]), abs(r__4)));
	if (s == 0.f) {
	    v[1].r = 0.f, v[1].i = 0.f;
	    v[2].r = 0.f, v[2].i = 0.f;
	} else {
	    i__1 = h_dim1 + 2;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h21s.r = q__1.r, h21s.i = q__1.i;
	    i__1 = (h_dim1 << 1) + 1;
	    q__2.r = h21s.r * h__[i__1].r - h21s.i * h__[i__1].i, q__2.i = 
		    h21s.r * h__[i__1].i + h21s.i * h__[i__1].r;
	    i__2 = h_dim1 + 1;
	    q__4.r = h__[i__2].r - s1->r, q__4.i = h__[i__2].i - s1->i;
	    i__3 = h_dim1 + 1;
	    q__6.r = h__[i__3].r - s2->r, q__6.i = h__[i__3].i - s2->i;
	    q__5.r = q__6.r / s, q__5.i = q__6.i / s;
	    q__3.r = q__4.r * q__5.r - q__4.i * q__5.i, q__3.i = q__4.r * 
		    q__5.i + q__4.i * q__5.r;
	    q__1.r = q__2.r + q__3.r, q__1.i = q__2.i + q__3.i;
	    v[1].r = q__1.r, v[1].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    q__4.r = h__[i__1].r + h__[i__2].r, q__4.i = h__[i__1].i + h__[
		    i__2].i;
	    q__3.r = q__4.r - s1->r, q__3.i = q__4.i - s1->i;
	    q__2.r = q__3.r - s2->r, q__2.i = q__3.i - s2->i;
	    q__1.r = h21s.r * q__2.r - h21s.i * q__2.i, q__1.i = h21s.r * 
		    q__2.i + h21s.i * q__2.r;
	    v[2].r = q__1.r, v[2].i = q__1.i;
	}
    } else {
	i__1 = h_dim1 + 1;
	q__2.r = h__[i__1].r - s2->r, q__2.i = h__[i__1].i - s2->i;
	q__1.r = q__2.r, q__1.i = q__2.i;
	i__2 = h_dim1 + 2;
	i__3 = h_dim1 + 3;
	s = (r__1 = q__1.r, abs(r__1)) + (r__2 = r_imag(&q__1), abs(r__2)) + (
		(r__3 = h__[i__2].r, abs(r__3)) + (r__4 = r_imag(&h__[h_dim1 
		+ 2]), abs(r__4))) + ((r__5 = h__[i__3].r, abs(r__5)) + (r__6 
		= r_imag(&h__[h_dim1 + 3]), abs(r__6)));
	if (s == 0.f) {
	    v[1].r = 0.f, v[1].i = 0.f;
	    v[2].r = 0.f, v[2].i = 0.f;
	    v[3].r = 0.f, v[3].i = 0.f;
	} else {
	    i__1 = h_dim1 + 2;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h21s.r = q__1.r, h21s.i = q__1.i;
	    i__1 = h_dim1 + 3;
	    q__1.r = h__[i__1].r / s, q__1.i = h__[i__1].i / s;
	    h31s.r = q__1.r, h31s.i = q__1.i;
	    i__1 = h_dim1 + 1;
	    q__4.r = h__[i__1].r - s1->r, q__4.i = h__[i__1].i - s1->i;
	    i__2 = h_dim1 + 1;
	    q__6.r = h__[i__2].r - s2->r, q__6.i = h__[i__2].i - s2->i;
	    q__5.r = q__6.r / s, q__5.i = q__6.i / s;
	    q__3.r = q__4.r * q__5.r - q__4.i * q__5.i, q__3.i = q__4.r * 
		    q__5.i + q__4.i * q__5.r;
	    i__3 = (h_dim1 << 1) + 1;
	    q__7.r = h__[i__3].r * h21s.r - h__[i__3].i * h21s.i, q__7.i = 
		    h__[i__3].r * h21s.i + h__[i__3].i * h21s.r;
	    q__2.r = q__3.r + q__7.r, q__2.i = q__3.i + q__7.i;
	    i__4 = h_dim1 * 3 + 1;
	    q__8.r = h__[i__4].r * h31s.r - h__[i__4].i * h31s.i, q__8.i = 
		    h__[i__4].r * h31s.i + h__[i__4].i * h31s.r;
	    q__1.r = q__2.r + q__8.r, q__1.i = q__2.i + q__8.i;
	    v[1].r = q__1.r, v[1].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = (h_dim1 << 1) + 2;
	    q__5.r = h__[i__1].r + h__[i__2].r, q__5.i = h__[i__1].i + h__[
		    i__2].i;
	    q__4.r = q__5.r - s1->r, q__4.i = q__5.i - s1->i;
	    q__3.r = q__4.r - s2->r, q__3.i = q__4.i - s2->i;
	    q__2.r = h21s.r * q__3.r - h21s.i * q__3.i, q__2.i = h21s.r * 
		    q__3.i + h21s.i * q__3.r;
	    i__3 = h_dim1 * 3 + 2;
	    q__6.r = h__[i__3].r * h31s.r - h__[i__3].i * h31s.i, q__6.i = 
		    h__[i__3].r * h31s.i + h__[i__3].i * h31s.r;
	    q__1.r = q__2.r + q__6.r, q__1.i = q__2.i + q__6.i;
	    v[2].r = q__1.r, v[2].i = q__1.i;
	    i__1 = h_dim1 + 1;
	    i__2 = h_dim1 * 3 + 3;
	    q__5.r = h__[i__1].r + h__[i__2].r, q__5.i = h__[i__1].i + h__[
		    i__2].i;
	    q__4.r = q__5.r - s1->r, q__4.i = q__5.i - s1->i;
	    q__3.r = q__4.r - s2->r, q__3.i = q__4.i - s2->i;
	    q__2.r = h31s.r * q__3.r - h31s.i * q__3.i, q__2.i = h31s.r * 
		    q__3.i + h31s.i * q__3.r;
	    i__3 = (h_dim1 << 1) + 3;
	    q__6.r = h21s.r * h__[i__3].r - h21s.i * h__[i__3].i, q__6.i = 
		    h21s.r * h__[i__3].i + h21s.i * h__[i__3].r;
	    q__1.r = q__2.r + q__6.r, q__1.i = q__2.i + q__6.i;
	    v[3].r = q__1.r, v[3].i = q__1.i;
	}
    }
    return;
} /* claqr1_ */

