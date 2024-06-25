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

/*  Definition: */
/*  =========== */

/*       SUBROUTINE CTPLQT( M, N, L, MB, A, LDA, B, LDB, T, LDT, WORK, */
/*                          INFO ) */

/*       INTEGER         INFO, LDA, LDB, LDT, N, M, L, MB */
/*       COMPLEX      A( LDA, * ), B( LDB, * ), T( LDT, * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > */
/* > CTPLQT computes a blocked LQ factorization of a complex */
/* > "triangular-pentagonal" matrix C, which is composed of a */
/* > triangular block A and pentagonal block B, using the compact */
/* > WY representation for Q. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix B, and the order of the */
/* >          triangular matrix A. */
/* >          M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix B. */
/* >          N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] L */
/* > \verbatim */
/* >          L is INTEGER */
/* >          The number of rows of the lower trapezoidal part of B. */
/* >          MIN(M,N) >= L >= 0.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] MB */
/* > \verbatim */
/* >          MB is INTEGER */
/* >          The block size to be used in the blocked QR.  M >= MB >= 1. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,M) */
/* >          On entry, the lower triangular M-by-M matrix A. */
/* >          On exit, the elements on and below the diagonal of the array */
/* >          contain the lower triangular matrix L. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[in,out] B */
/* > \verbatim */
/* >          B is COMPLEX array, dimension (LDB,N) */
/* >          On entry, the pentagonal M-by-N matrix B.  The first N-L columns */
/* >          are rectangular, and the last L columns are lower trapezoidal. */
/* >          On exit, B contains the pentagonal matrix V.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDB */
/* > \verbatim */
/* >          LDB is INTEGER */
/* >          The leading dimension of the array B.  LDB >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] T */
/* > \verbatim */
/* >          T is COMPLEX array, dimension (LDT,N) */
/* >          The lower triangular block reflectors stored in compact form */
/* >          as a sequence of upper triangular blocks.  See Further Details. */
/* > \endverbatim */
/* > */
/* > \param[in] LDT */
/* > \verbatim */
/* >          LDT is INTEGER */
/* >          The leading dimension of the array T.  LDT >= MB. */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          WORK is COMPLEX array, dimension (MB*M) */
/* > \endverbatim */
/* > */
/* > \param[out] INFO */
/* > \verbatim */
/* >          INFO is INTEGER */
/* >          = 0:  successful exit */
/* >          < 0:  if INFO = -i, the i-th argument had an illegal value */
/* > \endverbatim */

/*  Authors: */
/*  ======== */

/* > \author Univ. of Tennessee */
/* > \author Univ. of California Berkeley */
/* > \author Univ. of Colorado Denver */
/* > \author NAG Ltd. */

/* > \date June 2017 */

/* > \ingroup doubleOTHERcomputational */

/* > \par Further Details: */
/*  ===================== */
/* > */
/* > \verbatim */
/* > */
/* >  The input matrix C is a M-by-(M+N) matrix */
/* > */
/* >               C = [ A ] [ B ] */
/* > */
/* > */
/* >  where A is an lower triangular M-by-M matrix, and B is M-by-N pentagonal */
/* >  matrix consisting of a M-by-(N-L) rectangular matrix B1 on left of a M-by-L */
/* >  upper trapezoidal matrix B2: */
/* >          [ B ] = [ B1 ] [ B2 ] */
/* >                   [ B1 ]  <- M-by-(N-L) rectangular */
/* >                   [ B2 ]  <-     M-by-L lower trapezoidal. */
/* > */
/* >  The lower trapezoidal matrix B2 consists of the first L columns of a */
/* >  M-by-M lower triangular matrix, where 0 <= L <= MIN(M,N).  If L=0, */
/* >  B is rectangular M-by-N; if M=L=N, B is lower triangular. */
/* > */
/* >  The matrix W stores the elementary reflectors H(i) in the i-th row */
/* >  above the diagonal (of A) in the M-by-(M+N) input matrix C */
/* >            [ C ] = [ A ] [ B ] */
/* >                   [ A ]  <- lower triangular M-by-M */
/* >                   [ B ]  <- M-by-N pentagonal */
/* > */
/* >  so that W can be represented as */
/* >            [ W ] = [ I ] [ V ] */
/* >                   [ I ]  <- identity, M-by-M */
/* >                   [ V ]  <- M-by-N, same form as B. */
/* > */
/* >  Thus, all of information needed for W is contained on exit in B, which */
/* >  we call V above.  Note that V has the same form as B; that is, */
/* >            [ V ] = [ V1 ] [ V2 ] */
/* >                   [ V1 ] <- M-by-(N-L) rectangular */
/* >                   [ V2 ] <-     M-by-L lower trapezoidal. */
/* > */
/* >  The rows of V represent the vectors which define the H(i)'s. */
/* > */
/* >  The number of blocks is B = ceiling(M/MB), where each */
/* >  block is of order MB except for the last block, which is of order */
/* >  IB = M - (M-1)*MB.  For each of the B blocks, a upper triangular block */
/* >  reflector factor is computed: T1, T2, ..., TB.  The MB-by-MB (and IB-by-IB */
/* >  for the last block) T's are stored in the MB-by-N matrix T as */
/* > */
/* >               T = [T1 T2 ... TB]. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  ctplqt_(integer *m, integer *n, integer *l, integer *mb, 
	complex *a, integer *lda, complex *b, integer *ldb, complex *t, 
	integer *ldt, complex *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, t_dim1, t_offset, i__1, i__2, 
	    i__3, i__4;

    /* Local variables */
    integer i__, ib, lb, nb, iinfo;
    extern void  xerbla_(char *, integer *), ctprfb_(
	    char *, char *, char *, char *, integer *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, integer *, complex *, 
	    integer *, complex *, integer *, complex *, integer *), ctplqt2_(integer *, integer *, integer *,
	     complex *, integer *, complex *, integer *, complex *, integer *,
	     integer *);


/*  -- LAPACK computational routine (version 3.7.1) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */
/*     June 2017 */


/* ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    --work;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*l < 0 || *l > f2cmin(*m,*n) && f2cmin(*m,*n) >= 0) {
	*info = -3;
    } else if (*mb < 1 || *mb > *m && *m > 0) {
	*info = -4;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -6;
    } else if (*ldb < f2cmax(1,*m)) {
	*info = -8;
    } else if (*ldt < *mb) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CTPLQT", &i__1);
	return;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return;
    }

    i__1 = *m;
    i__2 = *mb;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {

/*     Compute the QR factorization of the current block */

/* Computing MIN */
	i__3 = *m - i__ + 1;
	ib = f2cmin(i__3,*mb);
/* Computing MIN */
	i__3 = *n - *l + i__ + ib - 1;
	nb = f2cmin(i__3,*n);
	if (i__ >= *l) {
	    lb = 0;
	} else {
	    lb = nb - *n + *l - i__ + 1;
	}

	ctplqt2_(&ib, &nb, &lb, &a[i__ + i__ * a_dim1], lda, &b[i__ + b_dim1],
		 ldb, &t[i__ * t_dim1 + 1], ldt, &iinfo);

/*     Update by applying H**T to B(I+IB:M,:) from the right */

	if (i__ + ib <= *m) {
	    i__3 = *m - i__ - ib + 1;
	    i__4 = *m - i__ - ib + 1;
	    ctprfb_("R", "N", "F", "R", &i__3, &nb, &ib, &lb, &b[i__ + b_dim1]
		    , ldb, &t[i__ * t_dim1 + 1], ldt, &a[i__ + ib + i__ * 
		    a_dim1], lda, &b[i__ + ib + b_dim1], ldb, &work[1], &i__4);
	}
    }
    return;

/*     End of CTPLQT */

} /* ctplqt_ */

