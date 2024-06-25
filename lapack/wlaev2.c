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

/* > \brief \b ZLAEV2 computes the eigenvalues and eigenvectors of a 2-by-2 symmetric/Hermitian matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLAEV2 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlaev2.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlaev2.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlaev2.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLAEV2( A, B, C, RT1, RT2, CS1, SN1 ) */

/*       DOUBLE PRECISION   CS1, RT1, RT2 */
/*       COMPLEX*16         A, B, C, SN1 */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLAEV2 computes the eigendecomposition of a 2-by-2 Hermitian matrix */
/* >    [  A         B  ] */
/* >    [  CONJG(B)  C  ]. */
/* > On return, RT1 is the eigenvalue of larger absolute value, RT2 is the */
/* > eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right */
/* > eigenvector for RT1, giving the decomposition */
/* > */
/* > [ CS1  CONJG(SN1) ] [    A     B ] [ CS1 -CONJG(SN1) ] = [ RT1  0  ] */
/* > [-SN1     CS1     ] [ CONJG(B) C ] [ SN1     CS1     ]   [  0  RT2 ]. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] A */
/* > \verbatim */
/* >          A is COMPLEX*16 */
/* >         The (1,1) element of the 2-by-2 matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] B */
/* > \verbatim */
/* >          B is COMPLEX*16 */
/* >         The (1,2) element and the conjugate of the (2,1) element of */
/* >         the 2-by-2 matrix. */
/* > \endverbatim */
/* > */
/* > \param[in] C */
/* > \verbatim */
/* >          C is COMPLEX*16 */
/* >         The (2,2) element of the 2-by-2 matrix. */
/* > \endverbatim */
/* > */
/* > \param[out] RT1 */
/* > \verbatim */
/* >          RT1 is DOUBLE PRECISION */
/* >         The eigenvalue of larger absolute value. */
/* > \endverbatim */
/* > */
/* > \param[out] RT2 */
/* > \verbatim */
/* >          RT2 is DOUBLE PRECISION */
/* >         The eigenvalue of smaller absolute value. */
/* > \endverbatim */
/* > */
/* > \param[out] CS1 */
/* > \verbatim */
/* >          CS1 is DOUBLE PRECISION */
/* > \endverbatim */
/* > */
/* > \param[out] SN1 */
/* > \verbatim */
/* >          SN1 is COMPLEX*16 */
/* >         The vector (CS1, SN1) is a unit right eigenvector for RT1. */
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
/* >  RT1 is accurate to a few ulps barring over/underflow. */
/* > */
/* >  RT2 may be inaccurate if there is massive cancellation in the */
/* >  determinant A*C-B*B; higher precision or correctly rounded or */
/* >  correctly truncated arithmetic would be needed to compute RT2 */
/* >  accurately in all cases. */
/* > */
/* >  CS1 and SN1 are accurate to a few ulps barring over/underflow. */
/* > */
/* >  Overflow is possible only if RT1 is within a factor of 5 of overflow. */
/* >  Underflow is harmless if the input data is 0 or exceeds */
/* >     underflow_threshold / macheps. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  wlaev2_(quadcomplex *a, quadcomplex *b, 
	quadcomplex *c__, quadreal *rt1, quadreal *rt2, quadreal *cs1,
	 quadcomplex *sn1)
{
    /* System generated locals */
    quadreal d__1, d__2, d__3;
    quadcomplex z__1, z__2;

    /* Local variables */
    quadreal t;
    quadcomplex w;
    extern void  qlaev2_(quadreal *, quadreal *, 
	    quadreal *, quadreal *, quadreal *, quadreal *, 
	    quadreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/* ===================================================================== */


    if (z_abs(b) == 0.) {
	w.r = 1., w.i = 0.;
    } else {
	d_cnjg(&z__2, b);
	d__1 = z_abs(b);
	z__1.r = z__2.r / d__1, z__1.i = z__2.i / d__1;
	w.r = z__1.r, w.i = z__1.i;
    }
    d__1 = a->r;
    d__2 = z_abs(b);
    d__3 = c__->r;
    qlaev2_(&d__1, &d__2, &d__3, rt1, rt2, cs1, &t);
    z__1.r = t * w.r, z__1.i = t * w.i;
    sn1->r = z__1.r, sn1->i = z__1.i;
    return;

/*     End of ZLAEV2 */

} /* wlaev2_ */

