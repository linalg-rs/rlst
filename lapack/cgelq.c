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

/* Table of constant values */

static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__2 = 2;


/*  Definition: */
/*  =========== */

/*       SUBROUTINE CGELQ( M, N, A, LDA, T, TSIZE, WORK, LWORK, */
/*                         INFO ) */

/*       INTEGER           INFO, LDA, M, N, TSIZE, LWORK */
/*       COMPLEX           A( LDA, * ), T( * ), WORK( * ) */


/* > \par Purpose: */
/*  ============= */
/* > */
/* > \verbatim */
/* > CGELQ computes a LQ factorization of an M-by-N matrix A. */
/* > \endverbatim */

/*  Arguments: */
/*  ========== */

/* > \param[in] M */
/* > \verbatim */
/* >          M is INTEGER */
/* >          The number of rows of the matrix A.  M >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in] N */
/* > \verbatim */
/* >          N is INTEGER */
/* >          The number of columns of the matrix A.  N >= 0. */
/* > \endverbatim */
/* > */
/* > \param[in,out] A */
/* > \verbatim */
/* >          A is COMPLEX array, dimension (LDA,N) */
/* >          On entry, the M-by-N matrix A. */
/* >          On exit, the elements on and below the diagonal of the array */
/* >          contain the M-by-f2cmin(M,N) lower trapezoidal matrix L */
/* >          (L is lower triangular if M <= N); */
/* >          the elements above the diagonal are used to store part of the */
/* >          data structure to represent Q. */
/* > \endverbatim */
/* > */
/* > \param[in] LDA */
/* > \verbatim */
/* >          LDA is INTEGER */
/* >          The leading dimension of the array A.  LDA >= f2cmax(1,M). */
/* > \endverbatim */
/* > */
/* > \param[out] T */
/* > \verbatim */
/* >          T is COMPLEX array, dimension (MAX(5,TSIZE)) */
/* >          On exit, if INFO = 0, T(1) returns optimal (or either minimal */
/* >          or optimal, if query is assumed) TSIZE. See TSIZE for details. */
/* >          Remaining T contains part of the data structure used to represent Q. */
/* >          If one wants to apply or construct Q, then one needs to keep T */
/* >          (in addition to A) and pass it to further subroutines. */
/* > \endverbatim */
/* > */
/* > \param[in] TSIZE */
/* > \verbatim */
/* >          TSIZE is INTEGER */
/* >          If TSIZE >= 5, the dimension of the array T. */
/* >          If TSIZE = -1 or -2, then a workspace query is assumed. The routine */
/* >          only calculates the sizes of the T and WORK arrays, returns these */
/* >          values as the first entries of the T and WORK arrays, and no error */
/* >          message related to T or WORK is issued by XERBLA. */
/* >          If TSIZE = -1, the routine calculates optimal size of T for the */
/* >          optimum performance and returns this value in T(1). */
/* >          If TSIZE = -2, the routine calculates minimal size of T and */
/* >          returns this value in T(1). */
/* > \endverbatim */
/* > */
/* > \param[out] WORK */
/* > \verbatim */
/* >          (workspace) COMPLEX array, dimension (MAX(1,LWORK)) */
/* >          On exit, if INFO = 0, WORK(1) contains optimal (or either minimal */
/* >          or optimal, if query was assumed) LWORK. */
/* >          See LWORK for details. */
/* > \endverbatim */
/* > */
/* > \param[in] LWORK */
/* > \verbatim */
/* >          LWORK is INTEGER */
/* >          The dimension of the array WORK. */
/* >          If LWORK = -1 or -2, then a workspace query is assumed. The routine */
/* >          only calculates the sizes of the T and WORK arrays, returns these */
/* >          values as the first entries of the T and WORK arrays, and no error */
/* >          message related to T or WORK is issued by XERBLA. */
/* >          If LWORK = -1, the routine calculates optimal size of WORK for the */
/* >          optimal performance and returns this value in WORK(1). */
/* >          If LWORK = -2, the routine calculates minimal size of WORK and */
/* >          returns this value in WORK(1). */
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

/* > \par Further Details */
/*  ==================== */
/* > */
/* > \verbatim */
/* > */
/* > The goal of the interface is to give maximum freedom to the developers for */
/* > creating any LQ factorization algorithm they wish. The triangular */
/* > (trapezoidal) L has to be stored in the lower part of A. The lower part of A */
/* > and the array T can be used to store any relevant information for applying or */
/* > constructing the Q factor. The WORK array can safely be discarded after exit. */
/* > */
/* > Caution: One should not expect the sizes of T and WORK to be the same from one */
/* > LAPACK implementation to the other, or even from one execution to the other. */
/* > A workspace query (for T and WORK) is needed at each execution. However, */
/* > for a given execution, the size of T and WORK are fixed and will not change */
/* > from one query to the next. */
/* > */
/* > \endverbatim */
/* > */
/* > \par Further Details particular to this LAPACK implementation: */
/*  ============================================================== */
/* > */
/* > \verbatim */
/* > */
/* > These details are particular for this LAPACK implementation. Users should not */
/* > take them for granted. These details may change in the future, and are unlikely not */
/* > true for another LAPACK implementation. These details are relevant if one wants */
/* > to try to understand the code. They are not part of the interface. */
/* > */
/* > In this version, */
/* > */
/* >          T(2): row block size (MB) */
/* >          T(3): column block size (NB) */
/* >          T(6:TSIZE): data structure needed for Q, computed by */
/* >                           CLASWLQ or CGELQT */
/* > */
/* >  Depending on the matrix dimensions M and N, and row and column */
/* >  block sizes MB and NB returned by ILAENV, CGELQ will use either */
/* >  CLASWLQ (if the matrix is short-and-wide) or CGELQT to compute */
/* >  the LQ factorization. */
/* > \endverbatim */
/* > */
/*  ===================================================================== */
void  cgelq_(integer *m, integer *n, complex *a, integer *lda, 
	complex *t, integer *tsize, complex *work, integer *lwork, integer *
	info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    integer mb, nb;
    logical mint, minw;
    integer nblcks;
    extern void  xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern void  cgelqt_(integer *, integer *, integer *, 
	    complex *, integer *, complex *, integer *, complex *, integer *);
    logical lminws, lquery;
    integer mintsz;
    extern void  claswlq_(integer *, integer *, integer *, 
	    integer *, complex *, integer *, complex *, integer *, complex *, 
	    integer *, integer *);


/*  -- LAPACK computational routine (version 3.7.0) -- */
/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd. -- */
/*     December 2016 */


/*  ===================================================================== */


/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --t;
    --work;

    /* Function Body */
    *info = 0;

    lquery = *tsize == -1 || *tsize == -2 || *lwork == -1 || *lwork == -2;

    mint = FALSE_;
    minw = FALSE_;
    if (*tsize == -2 || *lwork == -2) {
	if (*tsize != -1) {
	    mint = TRUE_;
	}
	if (*lwork != -1) {
	    minw = TRUE_;
	}
    }

/*     Determine the block size */

    if (f2cmin(*m,*n) > 0) {
	mb = ilaenv_(&c__1, "CGELQ ", " ", m, n, &c__1, &c_n1, (ftnlen)6, (
		ftnlen)1);
	nb = ilaenv_(&c__1, "CGELQ ", " ", m, n, &c__2, &c_n1, (ftnlen)6, (
		ftnlen)1);
    } else {
	mb = 1;
	nb = *n;
    }
    if (mb > f2cmin(*m,*n) || mb < 1) {
	mb = 1;
    }
    if (nb > *n || nb <= *m) {
	nb = *n;
    }
    mintsz = *m + 5;
    if (nb > *m && *n > *m) {
	if ((*n - *m) % (nb - *m) == 0) {
	    nblcks = (*n - *m) / (nb - *m);
	} else {
	    nblcks = (*n - *m) / (nb - *m) + 1;
	}
    } else {
	nblcks = 1;
    }

/*     Determine if the workspace size satisfies minimal size */

    lminws = FALSE_;
/* Computing MAX */
    i__1 = 1, i__2 = mb * *m * nblcks + 5;
    if ((*tsize < f2cmax(i__1,i__2) || *lwork < mb * *m) && *lwork >= *m && *
	    tsize >= mintsz && ! lquery) {
/* Computing MAX */
	i__1 = 1, i__2 = mb * *m * nblcks + 5;
	if (*tsize < f2cmax(i__1,i__2)) {
	    lminws = TRUE_;
	    mb = 1;
	    nb = *n;
	}
	if (*lwork < mb * *m) {
	    lminws = TRUE_;
	    mb = 1;
	}
    }

    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < f2cmax(1,*m)) {
	*info = -4;
    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = 1, i__2 = mb * *m * nblcks + 5;
	if (*tsize < f2cmax(i__1,i__2) && ! lquery && ! lminws) {
	    *info = -6;
	} else /* if(complicated condition) */ {
/* Computing MAX */
	    i__1 = 1, i__2 = *m * mb;
	    if (*lwork < f2cmax(i__1,i__2) && ! lquery && ! lminws) {
		*info = -8;
	    }
	}
    }

    if (*info == 0) {
	if (mint) {
	    t[1].r = (real) mintsz, t[1].i = 0.f;
	} else {
	    i__1 = mb * *m * nblcks + 5;
	    t[1].r = (real) i__1, t[1].i = 0.f;
	}
	t[2].r = (real) mb, t[2].i = 0.f;
	t[3].r = (real) nb, t[3].i = 0.f;
	if (minw) {
	    i__1 = f2cmax(1,*n);
	    work[1].r = (real) i__1, work[1].i = 0.f;
	} else {
/* Computing MAX */
	    i__2 = 1, i__3 = mb * *m;
	    i__1 = f2cmax(i__2,i__3);
	    work[1].r = (real) i__1, work[1].i = 0.f;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("CGELQ", &i__1);
	return;
    } else if (lquery) {
	return;
    }

/*     Quick return if possible */

    if (f2cmin(*m,*n) == 0) {
	return;
    }

/*     The LQ Decomposition */

    if (*n <= *m || nb <= *m || nb >= *n) {
	cgelqt_(m, n, &mb, &a[a_offset], lda, &t[6], &mb, &work[1], info);
    } else {
	claswlq_(m, n, &mb, &nb, &a[a_offset], lda, &t[6], &mb, &work[1], 
		lwork, info);
    }

/* Computing MAX */
    i__2 = 1, i__3 = mb * *m;
    i__1 = f2cmax(i__2,i__3);
    work[1].r = (real) i__1, work[1].i = 0.f;

    return;

/*     End of CGELQ */

} /* cgelq_ */

