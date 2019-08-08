/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/dev/DevHipRt.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Hip.hpp>

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace alpaka
{
    namespace event
    {
        class EventHipRt;
    }
}

namespace alpaka
{
    namespace queue
    {
        namespace hip
        {
            namespace detail
            {
                //#############################################################################
                //! The HIP RT blocking queue implementation.
                class QueueHipRtBlockingImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST QueueHipRtBlockingImpl(
                        dev::DevHipRt const & dev) :
                            m_dev(dev),
                            m_HipQueue()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device.
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // - hipStreamDefault: Default queue creation flag.
                        // - hipStreamNonBlocking: Specifies that work running in the created queue may run concurrently with work in queue 0 (the NULL queue),
                        //   and that the created queue should perform no implicit synchronization with queue 0.
                        // Create the queue on the current device.
                        // NOTE: hipStreamNonBlocking is required to match the semantic implemented in the alpaka CPU queue.
                        // It would be too much work to implement implicit default queue synchronization on CPU.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamCreateWithFlags(
                                &m_HipQueue,
                                hipStreamNonBlocking));
                    }
                    //-----------------------------------------------------------------------------
                    QueueHipRtBlockingImpl(QueueHipRtBlockingImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    QueueHipRtBlockingImpl(QueueHipRtBlockingImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtBlockingImpl const &) -> QueueHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(QueueHipRtBlockingImpl &&) -> QueueHipRtBlockingImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~QueueHipRtBlockingImpl()
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                        // Set the current device. \TODO: Is setting the current device before hipStreamDestroy required?
                        ALPAKA_HIP_RT_CHECK(
                            hipSetDevice(
                                m_dev.m_iDevice));
                        // In case the device is still doing work in the queue when hipStreamDestroy() is called, the function will return immediately
                        // and the resources associated with queue will be released automatically once the device has completed all work in queue.
                        // -> No need to synchronize here.
                        ALPAKA_HIP_RT_CHECK(
                            hipStreamDestroy(
                                m_HipQueue));
                    }

                public:
                    dev::DevHipRt const m_dev;   //!< The device this queue is bound to.
                    hipStream_t m_HipQueue;
                    //FIXME: workaround for failing stream query
                    int m_callees = 0;
                    std::mutex m_mutex;
                };
            } // detail
        } // hip

        //#############################################################################
        //! The HIP RT blocking queue.
        class QueueHipRtBlocking final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueHipRtBlocking(
                dev::DevHipRt const & dev) :
                m_spQueueImpl(std::make_shared<hip::detail::QueueHipRtBlockingImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            QueueHipRtBlocking(QueueHipRtBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueHipRtBlocking(QueueHipRtBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtBlocking const &) -> QueueHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueHipRtBlocking &&) -> QueueHipRtBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueHipRtBlocking const & rhs) const
            -> bool
            {
                return (m_spQueueImpl == rhs.m_spQueueImpl);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueHipRtBlocking const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST ~QueueHipRtBlocking() = default;

        public:
            std::shared_ptr<hip::detail::QueueHipRtBlockingImpl> m_spQueueImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueHipRtBlocking>
            {
                using type = dev::DevHipRt;
            };
            //#############################################################################
            //! The HIP RT blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueHipRtBlocking const & queue)
                -> dev::DevHipRt
                {
                    return queue.m_spQueueImpl->m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueHipRtBlocking>
            {
                using type = event::EventHipRt;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueHipRtBlocking,
                TTask>
            {
                //#############################################################################
                enum class CallbackState
                {
                    enqueued,
                    notified,
                    finished,
                };

                //#############################################################################
                struct CallbackSynchronizationData : public std::enable_shared_from_this<CallbackSynchronizationData>
                {
                    std::mutex m_mutex;
                    std::condition_variable m_event;
                    CallbackState state = CallbackState::enqueued;
                };

                //-----------------------------------------------------------------------------
                static void HIPRT_CB hipRtCallback(hipStream_t /*queue*/, hipError_t /*status*/, void *arg)
                {
                    // explicitly copy the shared_ptr so that this method holds the state even when the executing thread has already finished.
                    const auto pCallbackSynchronizationData = reinterpret_cast<CallbackSynchronizationData*>(arg)->shared_from_this();

                    // Notify the executing thread.
                    {
                        std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                        pCallbackSynchronizationData->state = CallbackState::notified;
                    }
                    pCallbackSynchronizationData->m_event.notify_one();

                    // Wait for the executing thread to finish the task if it has not already finished.
                    std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                    if(pCallbackSynchronizationData->state != CallbackState::finished)
                    {
                        pCallbackSynchronizationData->m_event.wait(
                            lock,
                            [pCallbackSynchronizationData](){
                                return pCallbackSynchronizationData->state == CallbackState::finished;
                            }
                        );
                    }
                }

                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueHipRtBlocking & queue,
                    TTask const & task)
                -> void
                {
                    {
                        // thread-safe callee incrementing
                        std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);
                        queue.m_spQueueImpl->m_callees += 1;
                    }

                    auto pCallbackSynchronizationData = std::make_shared<CallbackSynchronizationData>();

                    ALPAKA_HIP_RT_CHECK(hipStreamAddCallback(
                        queue.m_spQueueImpl->m_HipQueue,
                        hipRtCallback,
                        pCallbackSynchronizationData.get(),
                        0u));

                    // We start a new std::thread which stores the task to be executed.
                    // This circumvents the limitation that it is not possible to call HIP methods within the HIP callback thread.
                    // The HIP thread signals the std::thread when it is ready to execute the task.
                    // The HIP thread is waiting for the std::thread to signal that it is finished executing the task
                    // before it executes the next task in the queue (HIP stream).
                    std::thread t(
                        [pCallbackSynchronizationData, task, &queue](){

                            // thread-safe task execution and callee decrementing
                            std::lock_guard<std::mutex> guard(queue.m_spQueueImpl->m_mutex);

                            // If the callback has not yet been called, we wait for it.
                            {
                                std::unique_lock<std::mutex> lock(pCallbackSynchronizationData->m_mutex);
                                if(pCallbackSynchronizationData->state != CallbackState::notified)
                                {
                                    pCallbackSynchronizationData->m_event.wait(
                                        lock,
                                        [pCallbackSynchronizationData](){
                                            return pCallbackSynchronizationData->state == CallbackState::notified;
                                        }
                                    );
                                }

                                task();

                                // Notify the waiting HIP thread.
                                pCallbackSynchronizationData->state = CallbackState::finished;
                            }
                            pCallbackSynchronizationData->m_event.notify_one();

                            queue.m_spQueueImpl->m_callees -= 1;
                        }
                    );

                    t.join();
                }
            };
            //#############################################################################
            //! The HIP RT blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueHipRtBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // see: https://github.com/ROCm-Developer-Tools/HIP/blob/roc-1.9.x/tests/src/runtimeApi/stream/hipStreamWaitEvent.cpp

#if BOOST_COMP_HCC
                    // FIXME: workaround, see m_callees
                    return (queue.m_spQueueImpl->m_callees==0);
#else
                    // Query is allowed even for queues on non current device.
                    hipError_t ret = hipSuccess;
                    ALPAKA_HIP_RT_CHECK_IGNORE(
                        ret = hipStreamQuery(
                            queue.m_spQueueImpl->m_HipQueue),
                        hipErrorNotReady);
                    return (ret == hipSuccess);
#endif
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The HIP RT blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueHipRtBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueHipRtBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if BOOST_COMP_HCC
                    // FIXME: workaround, see m_callees
                    while(queue.m_spQueueImpl->m_callees>0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100u));
                    }
#else
                    // Sync is allowed even for queues on non current device.
                    ALPAKA_HIP_RT_CHECK( hipStreamSynchronize(
                        queue.m_spQueueImpl->m_HipQueue));
#endif
                }
            };
        }
    }
}

#endif
