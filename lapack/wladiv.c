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

/* > \brief \b ZLADIV performs complex division in doublereal arithmetic, avoiding unnecessary overflow. */

/*  =========== DOCUMENTATION =========== */

/* Online html documentation available at */
/*            http://www.netlib.org/lapack/explore-html/ */

/* > \htmlonly */
/* > Download ZLADIV + dependencies */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/zladiv.
f"> */
/* > [TGZ]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/zladiv.
f"> */
/* > [ZIP]</a> */
/* > <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/zladiv.
f"> */
/* > [TXT]</a> */
/* > \endhtmlonly */

/*  Definition: */
/*  =========== */

/*       COMPLEX*16     FUNCTION ZLADIV( X, Y ) */

/*       COMPLEX*16         X, Y */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > ZLADIV := X / Y, where X and Y are complex.  The computation of X / Y */
/* > will not overflow on an intermediary step unless the results */
/* > overflows. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] X */
/* > \verbatim */
/* >          X is COMPLEX*16 */
/* > \endverbatim */
/* > */
/* > \param[in] Y */
/* > \verbatim */
/* >          Y is COMPLEX*16 */
/* >          The complex scalars X and Y. */
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
/* Double Complex */ VOID wladiv_(quadcomplex * ret_val, quadcomplex *x, 
	quadcomplex *y)
{
    /* System generated locals */
    quadreal d__1, d__2, d__3, d__4;
    quadcomplex z__1;

    /* Local variables */
    quadreal zi, zr;
    extern void  qladiv_(quadreal *, quadreal *, 
	    quadreal *, quadreal *, quadreal *, quadreal *);


/*  -- LAPACK auxiliary routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     December 2016 */


/*  ===================================================================== */


    d__1 = x->r;
    d__2 = d_imag(x);
    d__3 = y->r;
    d__4 = d_imag(y);
    qladiv_(&d__1, &d__2, &d__3, &d__4, &zr, &zi);
    z__1.r = zr, z__1.i = zi;
     ret_val->r = z__1.r,  ret_val->i = z__1.i;

    return ;

/*     End of ZLADIV */

} /* wladiv_ */

