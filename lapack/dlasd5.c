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

#define __LAPACK_PRECISION_DOUBLE
#include "f2c.h"

/* > \brief \b DLASD5 computes the square root of the i-th eigenvalue of a positive symmetric rank-one modific
ation of a 2-by-2 diagonal matrix. Used by sbdsdc. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download DLASD5 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd5.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd5.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd5.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       SUBROUTINE DLASD5( I, D, Z, DELTA, RHO, DSIGMA, WORK ) */

/*       INTEGER            I */
/*       DOUBLE PRECISION   DSIGMA, RHO */
/*       DOUBLE PRECISION   D( 2 ), DELTA( 2 ), WORK( 2 ), Z( 2 ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > This subroutine computes the square root of the I-th eigenvalue */
/* > of a positive symmetric rank-one modification of a 2-by-2 diagonal */
/* > matrix */
/* > */
/* >            diag( D ) * diag( D ) +  RHO * Z * transpose(Z) . */
/* > */
/* > The diagonal entries in the array D are assumed to satisfy */
/* > */
/* >            0 <= D(i) < D(j)  for  i < j . */
/* > */
/* > We also assume RHO > 0 and that the Euclidean norm of the vector */
/* > Z is one. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] I */
/* > \verbatim */
/* >          I is INTEGER */
/* >         The index of the eigenvalue to be computed.  I = 1 or I = 2. */
/* > \endverbatim */
/* > */
/* > \param[in] D */
/* > \verbatim */
/* >          D is DOUBLE PRECISION array, dimension ( 2 ) */
/* >         The original eigenvalues.  We assume 0 <= D(1) < D(2). */
/* > \endverbatim */
/* > */
/* > \param[in] Z */
/* > \verbatim */
/* >          Z is DOUBLE PRECISION array, dimension ( 2 ) */
/* >         The components of the updating vector. */
/* > \endverbatim */
/* > */
/* > \param[out] DELTA */
/* > \verbatim */
/* >          DELTA is DOUBLE PRECISION array, dimension ( 2 ) */
/* >         Contains (D(j) - sigma_I) in its  j-th component. */
/* >         The vector DELTA contains the information necessary */
/* >         to construct the eigenvectors. */
/* > \endverbatim */
/* > */
/* > \param[in] RHO */
/* > \verbatim */
/* >          RHO is DOUBLE PRECISION */
/* >         The scalar in the symmetric updating formula. */
/* > \endverbatim */
/* > */
/* > \param[out] DSIGMA */
/* > \verbatim */
/* >          DSIGMA is DOUBLE PRECISION */
/* >         The computed sigma_I, the I-th updated eigenvalue. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is DOUBLE PRECISION array, dimension ( 2 ) */
/* >         WORK contains (D(j) + sigma_I) in its  j-th component. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup OTHERauxiliary */

/* > \par Contributors: */
/*  ================== */
/* > */
/* >     Ren-Cang Li, Computer Science Division, University of California */
/* >     at Berkeley, USA */
/* > */
/*  ===================================================================== */
void  dlasd5_(integer *i__, doublereal *d__, doublereal *z__, 
	doublereal *delta, doublereal *rho, doublereal *dsigma, doublereal *
	work)
{
    /* System generated locals */
    doublereal d__1;

    /* Local variables */
    doublereal b, c__, w, del, tau, delsq;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    /* Parameter adjustments */
    --work;
    --delta;
    --z__;
    --d__;

    /* Function Body */
    del = d__[2] - d__[1];
    delsq = del * (d__[2] + d__[1]);
    if (*i__ == 1) {
	w = *rho * 4. * (z__[2] * z__[2] / (d__[1] + d__[2] * 3.) - z__[1] * 
		z__[1] / (d__[1] * 3. + d__[2])) / del + 1.;
	if (w > 0.) {
	    b = delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[1] * z__[1] * delsq;

/*           B > ZERO, always */

/*           The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 ) */

	    tau = c__ * 2. / (b + M(sqrt)((d__1 = b * b - c__ * 4., abs(d__1))));

/*           The following TAU is DSIGMA - D( 1 ) */

	    tau /= d__[1] + M(sqrt)(d__[1] * d__[1] + tau);
	    *dsigma = d__[1] + tau;
	    delta[1] = -tau;
	    delta[2] = del - tau;
	    work[1] = d__[1] * 2. + tau;
	    work[2] = d__[1] + tau + d__[2];
/*           DELTA( 1 ) = -Z( 1 ) / TAU */
/*           DELTA( 2 ) = Z( 2 ) / ( DEL-TAU ) */
	} else {
	    b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[2] * z__[2] * delsq;

/*           The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	    if (b > 0.) {
		tau = c__ * -2. / (b + M(sqrt)(b * b + c__ * 4.));
	    } else {
		tau = (b - M(sqrt)(b * b + c__ * 4.)) / 2.;
	    }

/*           The following TAU is DSIGMA - D( 2 ) */

	    tau /= d__[2] + M(sqrt)((d__1 = d__[2] * d__[2] + tau, abs(d__1)));
	    *dsigma = d__[2] + tau;
	    delta[1] = -(del + tau);
	    delta[2] = -tau;
	    work[1] = d__[1] + tau + d__[2];
	    work[2] = d__[2] * 2. + tau;
/*           DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU ) */
/*           DELTA( 2 ) = -Z( 2 ) / TAU */
	}
/*        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) ) */
/*        DELTA( 1 ) = DELTA( 1 ) / TEMP */
/*        DELTA( 2 ) = DELTA( 2 ) / TEMP */
    } else {

/*        Now I=2 */

	b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	c__ = *rho * z__[2] * z__[2] * delsq;

/*        The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	if (b > 0.) {
	    tau = (b + M(sqrt)(b * b + c__ * 4.)) / 2.;
	} else {
	    tau = c__ * 2. / (-b + M(sqrt)(b * b + c__ * 4.));
	}

/*        The following TAU is DSIGMA - D( 2 ) */

	tau /= d__[2] + M(sqrt)(d__[2] * d__[2] + tau);
	*dsigma = d__[2] + tau;
	delta[1] = -(del + tau);
	delta[2] = -tau;
	work[1] = d__[1] + tau + d__[2];
	work[2] = d__[2] * 2. + tau;
/*        DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU ) */
/*        DELTA( 2 ) = -Z( 2 ) / TAU */
/*        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) ) */
/*        DELTA( 1 ) = DELTA( 1 ) / TEMP */
/*        DELTA( 2 ) = DELTA( 2 ) / TEMP */
    }
    return;

/*     End of DLASD5 */

} /* dlasd5_ */

