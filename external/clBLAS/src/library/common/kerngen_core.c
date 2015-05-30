/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


/*
 * Implementation of common logic for kernel
 * generators
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

#include <list.h>
#include <kerngen.h>
#include <mempat.h>

/*
 * TODO: Add checks for corruption for KgenContext and StatementBatch
 */

enum {
    TAB_WIDTH = 4,
};

struct KgenContext {
    char *buf;
    size_t bufLen;
    // name of the last declared function
    char *lastFname;
    size_t fnameLen;
    // current length without trailing '\0'
    size_t currLen;
    bool err;
    // current execution branch nesting
    int nesting;
    // number of tabs on the zero level of nesting
    int nrTabs;
    bool fmt;
};

struct StmtNode {
    char *stmt;
    ListNode node;
};

struct StatementBatch {
    ListHead statements[MAX_STATEMENT_PRIORITY + 1];
};

#ifdef TRACE_MALLOC

#define strdup(s)    strdupDebug(s)

static char
*strdupDebug(const char *s)
{
    char *dst;
    int len;

    len = strlen(s);
    dst = malloc(len + 1);
    if (dst != NULL) {
        memcpy(dst, s, len);
        dst[len] = '\0';
    }

    return dst;
}

#else                           /* TRACE_MALLOC */
#if defined(_MSC_VER)
#define strdup _strdup
#endif                          /* _MSC_VER */
#endif                          /* !TRACE_MALLOC */

static void
resetCtx(struct KgenContext *ctx)
{
    ctx->currLen = 0;
    ctx->nesting = 0;
    ctx->err = false;
    ctx->lastFname = NULL;
    ctx->fnameLen = 0;
    if (ctx->buf != NULL) {
        ctx->buf[0] = '\0';
    }
}

// extrace the first function name from a source buffer
static char*
searchFuncName(const char *source, size_t *len)
{
    char *sep;
    char *name = NULL;

    /*
     * Search the opening paranthesis. The word before it is
     * the function name
     */
    sep = strchr(source, '(');
    if (sep != NULL) {
        for (name = sep; name >= source; name--) {
            if ((*name == ' ') || (*name == '\n') || (*name == '*')) {
                break;
            }
        }
        name++;
        *len = (size_t)(sep - name);
    }

    return name;
}

/*
 * Immediately add string to source and does length check.
 *
 * The string should terminate with '\0' or pass size to copy
 */
static int
checkAddStr(struct KgenContext *ctx, const char *str, size_t slen)
{
    int ret = 0;
    size_t n = ctx->bufLen - ctx->currLen;
    size_t cplen;

    if (!slen) {
        slen = strlen(str);
        cplen = slen + 1;
    }
    else {
        cplen = slen;
    }

    if (ctx->buf == NULL) {
        ctx->currLen += slen;
    }
    else {
        if (cplen > n) {
            // make further code appendings unallowed
            ctx->err = true;
            ret = -1;
        }
        else {
            strncpy(ctx->buf + ctx->currLen, str, cplen);
            ctx->currLen += slen;
        }
    }

    return ret;
}


// add string to source, consiting of a prefix, a statement and a suffix
static int
addStr(
    struct KgenContext *ctx,
    const char *pref,
    const char *stmt,
    const char *suff)
{
    int ret = 0;
    char blank[MAX_NESTING * TAB_WIDTH];
    int i;
    char *sep = NULL;
    size_t len = 0;
    const int nblanks = (ctx->nesting + ctx->nrTabs) * TAB_WIDTH;

    if (nblanks && ctx->fmt) {
        for (i = 0; i < nblanks; i++) {
            blank[i] = ' ';
        }

        /*
         *  add formatting symbols if there is a prefix,
         *  or the statement don't begin with the new line
         *  symbols
         */
        if (pref || (stmt && (stmt[0] != '\n'))) {
            ret = checkAddStr(ctx, blank, nblanks);
        }
    }

    if (!ret && pref) {
        ret = checkAddStr(ctx, pref, 0);
    }

    /*
     * add the statement itself,
     * format the multiline ones if it's needed.
     */
    while (!ret && stmt) {
        if (ctx->fmt) {
            /*
             * do not add tabulation for lines consisting of
             * the new line symbol only
             */
            if (*stmt != '\n') {
                if (sep && nblanks) {
                    ret = checkAddStr(ctx, blank, nblanks);
                    if (ret) {
                        break;
                    }
                }
                sep = strchr(stmt, '\n');
                // skip the new line symbol if it is at the end of the line
                if (sep && (sep[1] == '\0')) {
                    sep = NULL;
                }
                len = (sep) ? (sep - stmt + 1) : 0;
            }
            else {
                /*
                 * The line can start with the new line symbol
                 * and have not any prefix. The assignment
                 * ensures the tabulation for the case.
                 */
                sep = (sep) ? sep : ((char*)stmt);
                len = (stmt[1] == '\0') ? 0 : 1;
            }
        }
        ret = checkAddStr(ctx, stmt, len);
        if (len) {
            stmt += len;
        }
        else {
            stmt = NULL;
        }
    }

    if (!ret && suff) {
        ret = checkAddStr(ctx, suff, 0);
    }

    return ret;
}

struct KgenContext
*createKgenContext(char *srcBuf, size_t srcBufLen, bool fmt)
{
    struct KgenContext *ctx;

    ctx = malloc(sizeof(struct KgenContext));
    if (ctx != NULL) {
        ctx->buf = srcBuf;
        ctx->bufLen = srcBufLen;
        ctx->fmt = fmt;
        ctx->nrTabs = 0;
        resetCtx(ctx);
    }

    return ctx;
}

static void
flushDestroyStmtNode(ListNode *l, void *priv)
{
    struct StmtNode *snode = container_of(l, node, struct StmtNode);

    if (priv != NULL) {
        addStr((struct KgenContext*)priv, NULL, snode->stmt, NULL);
    }
    free(snode->stmt);
    free(snode);
}

void
destroyKgenContext(struct KgenContext *ctx)
{
    if (ctx->lastFname) {
        free(ctx->lastFname);
    }
    free(ctx);
}

void
resetKgenContext(struct KgenContext *ctx)
{
    if (ctx->lastFname) {
        free(ctx->lastFname);
    }
    resetCtx(ctx);
}

int
kgenSyncFormatting(
    struct KgenContext *srcCtx,
    const struct KgenContext *dstCtx,
    int nrTabs)
{
    int ret = -EINVAL;

    if (nrTabs >= 0 && (nrTabs + dstCtx->nesting <= MAX_TABS)) {
        srcCtx->nesting = nrTabs + dstCtx->nesting;
        ret = 0;
    }

    return ret;
}

int
kgenDeclareFunction(struct KgenContext *ctx, const char *decl)
{
    int ret;
    size_t len;
    char *dbuf;
    const char *fnName;

    if (ctx->err || ctx->nesting) {
        ctx->err = true;
        return -1;
    }
    else {
        fnName = searchFuncName(decl, &len);
        if (fnName == NULL) {
            ret = -1;
        }
        else {
            // save the last declaration without
            dbuf = ctx->lastFname;
            if (dbuf == NULL) {
                dbuf = malloc(len + 1);
            }
            else if (ctx->fnameLen < len + 1) {
                dbuf = realloc(ctx->lastFname, len + 1);
                ctx->fnameLen = len + 1;
            }

            if (dbuf == NULL) {
                ret = -1;
            }
            else {
                strncpy(dbuf, fnName, len);
                dbuf[len] = '\0';
                ctx->lastFname = dbuf;
                ret = addStr(ctx, NULL, decl, NULL);
            }
        }

        if (ret) {
            ctx->err = true;
        }
    }

    return ret;
}

int
kgenBeginFuncBody(struct KgenContext *ctx)
{
    int ret;

    if (ctx->err || ctx->nesting) {
        ctx->err = true;
        ret = -1;
    }
    else {
        ret = addStr(ctx, NULL, NULL, "{\n");
        if (!ret) {
            ctx->nesting++;
        }
    }

    return ret;
}

int
kgenEndFuncBody(struct KgenContext *ctx)
{
    int ret;

    if (ctx->err || (ctx->nesting != 1)) {
        ctx->err = true;
        ret = -1;
    }
    else {
        ctx->nesting--;
        ret = addStr(ctx, NULL, NULL, "}\n");
    }

    return ret;
}

int
kgenGetLastFuncName(
    char *buf,
    size_t buflen,
    const struct KgenContext *ctx)
{
    size_t len;
    int ret = -1;

    if (ctx->lastFname) {
        len = strlen(ctx->lastFname);
        if (buflen >= len + 1) {
            strncpy(buf, ctx->lastFname, len);
            buf[len] = '\0';
            ret = 0;
        }
    }

    return ret;
}

int
kgenBeginBranch(struct KgenContext *ctx, const char *stmt)
{
    int ret;

    if (ctx->err || (ctx->nesting == MAX_NESTING)) {
        ctx->err = true;
        ret = -1;
    }
    else {
        const char *suff;

        if (stmt == NULL) {
            stmt = "";
            suff = "{\n";
        }
        else {
            suff = " {\n";
        }

        ret = addStr(ctx, NULL, stmt, suff);
        if (!ret) {
            ctx->nesting++;
        }
    }

    return ret;
}


int
kgenEndBranch(struct KgenContext *ctx, const char *stmt)
{
    const char *pref;
    const char *suff;

    if (ctx->err || !ctx->nesting) {
        ctx->err = true;
        return -1;
    }

    ctx->nesting--;

    if (stmt) {
        pref = "} ";
        suff = ";\n";
    }
    else {
        pref = "}\n";
        suff = NULL;
    }

    return addStr(ctx, pref, stmt, suff);
}

int
kgenAddStmt(struct KgenContext *ctx, const char *stmt)
{
    int ret = 0;

    if (ctx->err) {
        ret = -1;
    }
    else if (stmt != NULL) {
        ret = addStr(ctx, NULL, stmt, NULL);
    }

    return ret;
}

int
kgenPrintf(struct KgenContext *ctx, const char *fmt,...)
{
    char buf[MAX_STATEMENT_LENGTH];
    va_list ap;
    int len;

    if (ctx->err) {
        return -1;
    }

    va_start(ap, fmt);
    len = vsnprintf(buf, MAX_STATEMENT_LENGTH, fmt, ap);
    va_end(ap);

    if (len >= MAX_STATEMENT_LENGTH) {  /* has the statement been truncated? */
        return -1;
    }

    return addStr(ctx, NULL, buf, NULL);
}

struct StatementBatch
*createStmtBatch(void)
{
    struct StatementBatch *batch;

    batch = malloc(sizeof(struct StatementBatch));
    if (batch != NULL) {
        int i;

        for (i = 0; i <= MAX_STATEMENT_PRIORITY; i++) {
            listInitHead(&batch->statements[i]);
        }
    }

    return batch;
}

int
kgenAddStmtToBatch(
    struct StatementBatch *batch,
    int priority,
    const char *stmt)
{
    struct StmtNode *snode;
    int ret = -ENOMEM;

    if (priority == MAX_STATEMENT_PRIORITY) {
        return -EINVAL;
    }

    snode = malloc(sizeof(struct StmtNode));
    if (snode != NULL) {
        snode->stmt = strdup(stmt);
        if (snode->stmt != NULL) {
            listAddToTail(&batch->statements[priority], &snode->node);
            ret = 0;
        }
        else {
            free(snode);
        }
    }

    return ret;
}

int
kgenBatchPrintf(
    struct StatementBatch *batch,
    int priority,
    const char *fmt,...)
{
    char buf[MAX_STATEMENT_LENGTH];
    va_list ap;
    int len;

    va_start(ap, fmt);
    len = vsnprintf(buf, MAX_STATEMENT_LENGTH, fmt, ap);
    va_end(ap);

    if (len >= MAX_STATEMENT_LENGTH) {  /* has the statement been truncated? */
        return -1;
    }

    kgenAddStmtToBatch(batch, priority, buf);

    return 0;
}

int
flushStmtBatch(struct KgenContext *ctx, struct StatementBatch *batch)
{
    int i = 0;

    for (i = 0; i <= MAX_STATEMENT_PRIORITY; i++) {
        listDoForEachPrivSafe(&batch->statements[i], flushDestroyStmtNode, ctx);
        listInitHead(&batch->statements[i]);
    }

    return (ctx->err) ? -1 : 0;
}

void
destroyStmtBatch(struct StatementBatch *batch)
{
    int i;

    for (i = 0; i <= MAX_STATEMENT_PRIORITY; i++) {
       listDoForEachPrivSafe(&batch->statements[i], flushDestroyStmtNode, NULL);
    }
    free(batch);
}

int
kgenAddBlankLine(struct KgenContext *ctx)
{
    int ret;

    if (ctx->err) {
        ret = -1;
    }
    else {
        ret = addStr(ctx, NULL, NULL, "\n");
    }

    return ret;
}

size_t
kgenSourceSize(struct KgenContext *ctx)
{
    return ctx->currLen;
}
