import tensorflow as tf
import os


def filter_events(input_path, output_path, max_step):
    """
    Filter out events with 'step' greater than max_step from a TensorBoard events file.

    Parameters:
    - input_path: Path to the original TensorBoard events file.
    - output_path: Path where the filtered events file will be saved.
    - max_step: Maximum step number. Events with 'step' greater than this will be removed.
    """

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    writer = tf.summary.create_file_writer(output_path)
    step = 0
    for e in tf.compat.v1.train.summary_iterator(input_path):

        # The 'step' is available as 'e.step' for each event
        if e.step <= max_step:
            step = max(step, e.step)
            # Use the writer to write the event if its step is within the desired range
            with writer.as_default():
                tf.summary.experimental.write_raw_pb(e.SerializeToString(), step=e.step)

    print(step)
    writer.close()


# Example usage
input_path = "/home/ofri.masad/events.out.tfevents.1708816134.deci-tzag6-desktop.3119789.0"
output_path = "/home/ofri.masad/events.out.tfevents.1708816134.deci-tzag6-desktop.3119789.1"
max_step = 199  # Replace 1000 with your desired maximum step

filter_events(input_path, output_path, max_step)
