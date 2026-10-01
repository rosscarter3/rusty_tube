//! Thread scheduling hints.
//!
//! On macOS a thread with no quality-of-service class can be scheduled on
//! the efficiency cores, and a background or occluded app is throttled
//! further; the simulation then runs several times slower than it can.
//! Marking compute threads user-initiated keeps them on the performance
//! cores. A no-op elsewhere.

#[cfg(target_os = "macos")]
pub fn user_initiated() {
    // qos_class_t QOS_CLASS_USER_INITIATED = 0x19 (sys/qos.h).
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        pthread_set_qos_class_self_np(0x19, 0);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn user_initiated() {}
