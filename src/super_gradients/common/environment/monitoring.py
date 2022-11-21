# import psutil
#
#
# def machine_stats():
#     """
#     :return: machine stats dictionary, all values expressed in megabytes
#     """
#     cpu_usage = [float(v) for v in psutil.cpu_percent(percpu=True)]
#     stats = {
#         "cpu_usage": sum(cpu_usage) / float(len(cpu_usage)),
#     }
#
#     bytes_per_megabyte = 1024 ** 2
#
#     def bytes_to_megabytes(x):
#         return x / bytes_per_megabyte
#
#     virtual_memory = psutil.virtual_memory()
#     # stats["memory_used_gb"] = bytes_to_megabytes(virtual_memory.used) / 1024
#     stats["memory_used_gb"] = bytes_to_megabytes(
#         self._get_process_used_memory() if self._process_info else virtual_memory.used) / 1024
#     stats["memory_free_gb"] = bytes_to_megabytes(virtual_memory.available) / 1024
#     disk_use_percentage = psutil.disk_usage(Text(Path.home())).percent
#     stats["disk_free_percent"] = 100.0 - disk_use_percentage
#     with warnings.catch_warnings():
#         if logging.root.level > logging.DEBUG:  # If the logging level is bigger than debug, ignore
#             # psutil.sensors_temperatures warnings
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#         sensor_stat = (psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {})
#     if "coretemp" in sensor_stat and len(sensor_stat["coretemp"]):
#         stats["cpu_temperature"] = max([float(t.current) for t in sensor_stat["coretemp"]])
#
#     # protect against permission issues
#     # update cached measurements
#     # noinspection PyBroadException
#     try:
#         net_stats = psutil.net_io_counters()
#         stats["network_tx_mbs"] = bytes_to_megabytes(net_stats.bytes_sent)
#         stats["network_rx_mbs"] = bytes_to_megabytes(net_stats.bytes_recv)
#     except Exception:
#         pass
#
#     # protect against permission issues
#     # noinspection PyBroadException
#     try:
#         io_stats = psutil.disk_io_counters()
#         stats["io_read_mbs"] = bytes_to_megabytes(io_stats.read_bytes)
#         stats["io_write_mbs"] = bytes_to_megabytes(io_stats.write_bytes)
#     except Exception:
#         pass
#
#     # check if we can access the gpu statistics
#     if self._gpustat:
#         # noinspection PyBroadException
#         try:
#             stats.update(self._get_gpu_stats())
#         except Exception:
#             # something happened and we can't use gpu stats,
#             self._gpustat_fail += 1
#             if self._gpustat_fail >= 3:
#                 self._task.get_logger().report_text('ClearML Monitor: GPU monitoring failed getting GPU reading, '
#                                                     'switching off GPU monitoring')
#                 self._gpustat = None
#
#     return stats
