/* A two-function threading shim, so the motion search runs on Windows too.
 *
 * mingw-w64 ships winpthreads and `-pthread` would work, but that pulls a DLL
 * dependency into what is meant to be a self-contained executable. The search
 * needs exactly threads, one mutex and a core count, so the Win32 versions of
 * those three are cheaper than the dependency.
 */

#ifndef HVE_THREAD_H
#define HVE_THREAD_H

#ifdef _WIN32

/* Keep windows.h from dragging in the rest of the platform and from defining
 * min/max as macros, which breaks any C that uses those as identifiers. */
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

typedef HANDLE hve_thread;
typedef CRITICAL_SECTION hve_mutex;

static inline int hve_thread_start(hve_thread *t, unsigned (__stdcall *fn)(void *),
                                   void *arg)
{
    *t = (HANDLE)CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)fn, arg, 0, NULL);
    return *t ? 0 : -1;
}

static inline void hve_thread_join(hve_thread t)
{
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}

static inline void hve_mutex_init(hve_mutex *m) { InitializeCriticalSection(m); }
static inline void hve_mutex_lock(hve_mutex *m) { EnterCriticalSection(m); }
static inline void hve_mutex_unlock(hve_mutex *m) { LeaveCriticalSection(m); }
static inline void hve_mutex_free(hve_mutex *m) { DeleteCriticalSection(m); }

#define HVE_WORKER(name, arg) unsigned __stdcall name(void *arg)
#define HVE_WORKER_RETURN return 0

static inline int hve_cpu_count(void)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors < 1 ? 1 : (int)si.dwNumberOfProcessors;
}

#else /* POSIX */

#include <pthread.h>
#include <unistd.h>

typedef pthread_t hve_thread;
typedef pthread_mutex_t hve_mutex;

static inline int hve_thread_start(hve_thread *t, void *(*fn)(void *), void *arg)
{
    return pthread_create(t, NULL, fn, arg);
}

static inline void hve_thread_join(hve_thread t) { pthread_join(t, NULL); }
static inline void hve_mutex_init(hve_mutex *m) { pthread_mutex_init(m, NULL); }
static inline void hve_mutex_lock(hve_mutex *m) { pthread_mutex_lock(m); }
static inline void hve_mutex_unlock(hve_mutex *m) { pthread_mutex_unlock(m); }
static inline void hve_mutex_free(hve_mutex *m) { pthread_mutex_destroy(m); }

#define HVE_WORKER(name, arg) void *name(void *arg)
#define HVE_WORKER_RETURN return NULL

static inline int hve_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n < 1 ? 1 : (int)n;
}

#endif

#endif /* HVE_THREAD_H */
