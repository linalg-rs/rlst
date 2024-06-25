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

/* > \brief \b SLAPY3 returns M(sqrt)(x2+y2+z2). */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download SLAPY3 + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slapy3.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slapy3.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slapy3.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       REAL             FUNCTION SLAPY3( X, Y, Z ) */

/*       REAL               X, Y, Z */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > SLAPY3 returns M(sqrt)(x**2+y**2+z**2), taking care not to cause */
/* > unnecessary overflow. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] X */
/* > \verbatim */
/* >          X is REAL */
/* > \endverbatim */
/* > */
/* > \param[in] Y */
/* > \verbatim */
/* >          Y is REAL */
/* > \endverbatim */
/* > */
/* > \param[in] Z */
/* > \verbatim */
/* >          Z is REAL */
/* >          X, Y and Z specify the values x, y and z. */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date December 2016 */

/* > \ingroup OTHERauxiliary */

/*  ===================================================================== */
real slapy3_(real *x, real *y, real *z__)
{
    /* System generated locals */
    real ret_val, r__1, r__2, r__3;

    /* Local variables */
    real w, xabs, yabs, zabs;


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    xabs = abs(*x);
    yabs = abs(*y);
    zabs = abs(*z__);
/* Computing MAX */
    r__1 = f2cmax(xabs,yabs);
    w = f2cmax(r__1,zabs);
    if (w == 0.f) {
/*     W can be zero for f2cmax(0,nan,0) */
/*     adding all three entries together will make sure */
/*     NaN will not disappear. */
	ret_val = xabs + yabs + zabs;
    } else {
/* Computing 2nd power */
	r__1 = xabs / w;
/* Computing 2nd power */
	r__2 = yabs / w;
/* Computing 2nd power */
	r__3 = zabs / w;
	ret_val = w * M(sqrt)(r__1 * r__1 + r__2 * r__2 + r__3 * r__3);
    }
    return ret_val;

/*     End of SLAPY3 */

} /* slapy3_ */

