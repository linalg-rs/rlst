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

static halfcomplex c_b1 = {1.,0.};
static halfcomplex c_b2 = {0.,0.};
static integer c__1 = 1;

/* > \brief \b ZLARF applies an elementary reflector to a general rectangular matrix. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLARF + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zlarf.f
"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zlarf.f
"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zlarf.f
"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE ZLARF( SIDE, M, N, V, INCV, TAU, C, LDC, WORK ) */

/*       CHARACTER          SIDE */
/*       INTEGER            INCV, LDC, M, N */
/*       COMPLEX*16         TAU */
/*       COMPLEX*16         C( LDC, * ), V( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLARF applies a complex elementary reflector H to a complex M-by-N */
/* > matrix C, from either the left or the right. H is represented in the */
/* > form */
/* > */
/* >       H = I - tau * v * v**H */
/* > */
/* > where tau is a complex scalar and v is a complex vector. */
/* > */
/* > If tau = 0, then H is taken to be the unit matrix. */
/* > */
/* > To apply H**H, supply conjg(tau) instead */
/* > tau. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] SIDE */
/* > \verbatim */
/* >          SIDE is CHARACTER*1 */
/* >          = 'L': form  H * C */
/* >          = 'R': form  C * H */
/* > \endverbatim */
/* > */
/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix C. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix C. */
/* > \endverbatim */
/* > */
/* > \param[in] V */
/* > \verbatim */
/* >          V is COMPLEX*16 array, dimension */
/* >                     (1 + (M-1)*abs(INCV)) if SIDE = 'L' */
/* >                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R' */
/* >          The vector v in the representation of H. V is not used if */
/* >          TAU = 0. */
/* > \endverbatim */
/* > */
/* > \param[in] INCV */
/* > \verbatim */
/* >          INCV is INTEGER */
/* >          The increment between elements of v. INCV <> 0. */
/* > \endverbatim */
/* > */
/* > \param[in] TAU */
/* > \verbatim */
/* >          TAU is COMPLEX*16 */
/* >          The value tau in the representation of H. */
/* > \endverbatim */
/* > */
/* > \param[in,out] C */
/* > \verbatim */
/* >          C is COMPLEX*16 array, dimension (LDC,N) */
/* >          On entry, the M-by-N matrix C. */
/* >          On exit, C is overwritten by the matrix H * C if SIDE = 'L', */
/* >          or C * H if SIDE = 'R'. */
/* > \endverbatim */
/* > */
/* > \param[in] LDC */
/* > \verbatim */
/* >          LDC is INTEGER */
/* >          The leading dimension of the array C. LDC >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX*16 array, dimension */
/* >                         (N) if SIDE = 'L' */
/* >                      or (M) if SIDE = 'R' */
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
void  klarf_(char *side, integer *m, integer *n, halfcomplex 
	*v, integer *incv, halfcomplex *tau, halfcomplex *c__, integer *
	ldc, halfcomplex *work)
{
    /* System generated locals */
    integer c_dim1, c_offset, i__1;
    halfcomplex z__1;

    /* Local variables */
    integer i__;
    logical applyleft;
    extern logical lsame_(char *, char *);
    integer lastc;
    extern void  kgerc_(integer *, integer *, halfcomplex *, 
	    halfcomplex *, integer *, halfcomplex *, integer *, 
	    halfcomplex *, integer *), kgemv_(char *, integer *, integer *, 
	    halfcomplex *, halfcomplex *, integer *, halfcomplex *, 
	    integer *, halfcomplex *, halfcomplex *, integer *);
    integer lastv;
    extern integer ilaklc_(integer *, integer *, halfcomplex *, integer *), 
	    ilaklr_(integer *, integer *, halfcomplex *, integer *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    applyleft = lsame_(side, "L");
    lastv = 0;
    lastc = 0;
    if (tau->r != 0. || tau->i != 0.) {
/*     Set up variables for scanning V.  LASTV begins pointing to the end */
/*     of V. */
	if (applyleft) {
	    lastv = *m;
	} else {
	    lastv = *n;
	}
	if (*incv > 0) {
	    i__ = (lastv - 1) * *incv + 1;
	} else {
	    i__ = 1;
	}
/*     Look for the last non-zero row in V. */
	for(;;) { /* while(complicated condition) */
	    i__1 = i__;
	    if (!(lastv > 0 && (v[i__1].r == 0. && v[i__1].i == 0.)))
	    	break;
	    --lastv;
	    i__ -= *incv;
	}
	if (applyleft) {
/*     Scan for the last non-zero column in C(1:lastv,:). */
	    lastc = ilaklc_(&lastv, n, &c__[c_offset], ldc);
	} else {
/*     Scan for the last non-zero row in C(:,1:lastv). */
	    lastc = ilaklr_(m, &lastv, &c__[c_offset], ldc);
	}
    }
/*     Note that lastc.eq.0 renders the BLAS operations null; no special */
/*     case is needed at this level. */
    if (applyleft) {

/*        Form  H * C */

	if (lastv > 0) {

/*           w(1:lastc,1) := C(1:lastv,1:lastc)**H * v(1:lastv,1) */

	    kgemv_("Conjugate transpose", &lastv, &lastc, &c_b1, &c__[
		    c_offset], ldc, &v[1], incv, &c_b2, &work[1], &c__1);

/*           C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**H */

	    z__1.r = -tau->r, z__1.i = -tau->i;
	    kgerc_(&lastv, &lastc, &z__1, &v[1], incv, &work[1], &c__1, &c__[
		    c_offset], ldc);
	}
    } else {

/*        Form  C * H */

	if (lastv > 0) {

/*           w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1) */

	    kgemv_("No transpose", &lastc, &lastv, &c_b1, &c__[c_offset], ldc,
		     &v[1], incv, &c_b2, &work[1], &c__1);

/*           C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**H */

	    z__1.r = -tau->r, z__1.i = -tau->i;
	    kgerc_(&lastc, &lastv, &z__1, &work[1], &c__1, &v[1], incv, &c__[
		    c_offset], ldc);
	}
    }
    return;

/*     End of ZLARF */

} /* klarf_ */

