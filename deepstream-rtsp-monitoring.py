#!/usr/bin/env python3
import logging
import sys

logging.basicConfig(level=logging.INFO)

# NVIDIA DeepStream 6.3 installation path
sys.path.append("/opt/nvidia/deepstream/deepstream/sources/apps/deepstream_python_apps/apps/")

from common.bus_call import bus_call
import pyds
import math
import logging
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib
import datetime

import argparse

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_CELLPHONE = 67

MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 720
TILED_OUTPUT_HEIGHT = 1280
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0

logger = logging.getLogger()

# pgie_src_pad_buffer_probe  will extract metadata received on OSD sink pad
# and update params for drawing rectangle, object information etc.
def pgie_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        logger.error("Unable to get GstBuffer")
        return

    phone_count = 0

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.class_id == PGIE_CLASS_ID_CELLPHONE:
                obj_meta.text_params.display_text = "Cellphone is detected!"
                phone_count = phone_count + 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        logger.info(f"Frame Number={frame_number}; Number of Objects={num_rects}; PHONE_count={phone_count}")

        if ts_from_rtsp:
            ts = frame_meta.ntp_timestamp / 1000000000  # Retrieve timestamp, put decimal in proper position for Unix format
            formatted_timestamp = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            logger.info("RTSP Timestamp: %s", formatted_timestamp)  # Convert timestamp to UTC

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    logger.info("gstname= %s", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        logger.info("features= %s", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.error("Failed to link decoder src pad to source bin ghost pad")
        else:
            logger.error("Error: Decodebin did not pick nvidia decoder plugin.")


def decodebin_child_added(child_proxy, element, name, user_data):
    logger.info("Decodebin child added: %s", name)
    if name.find("decodebin") != -1:
        element.connect("child-added", decodebin_child_added, user_data)

    if ts_from_rtsp and name.find("source") != -1:
        pyds.configure_source_for_ntp_sync(hash(element))


def create_source_bin(uri):
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-0"
    logger.info(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error("Unable to create source bin")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        logger.error(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target(
            "src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.error("Failed to add ghost pad in source bin")
        return None
    return nbin


def main(args):
    # Check input arguments
    number_sources = len(args)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    logger.info("Creating Pipeline")
    pipeline = Gst.Pipeline()

    if not pipeline:
        logger.error("Unable to create Pipeline")
    logger.info("Creating streamux")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        logger.error("Unable to create NvStreamMux")

    pipeline.add(streammux)
    logger.info("Creating source_bin_0")
    uri_name = args[0]
    source_bin = create_source_bin(uri_name)
    if not source_bin:
        logger.error("Unable to create source bin")
    pipeline.add(source_bin)
    padname = "sink_0"
    sinkpad = streammux.get_request_pad(padname)
    if not sinkpad:
        logger.error("Unable to create sink pad bin")
    srcpad = source_bin.get_static_pad("src")
    if not srcpad:
        logger.error("Unable to create src pad bin")
    srcpad.link(sinkpad)

    logger.info("Creating Pgie")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        logger.error("Unable to create pgie")
    logger.info("Creating tiler")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        logger.error("Unable to create tiler")
    logger.info("Creating nvvidconv")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        logger.error("Unable to create nvvidconv")
    logger.info("Creating nvosd")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        logger.error("Unable to create nvosd")
    nvvidconv_postosd = Gst.ElementFactory.make(
        "nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        logger.error("Unable to create nvvidconv_postosd")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")
    )

    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        logger.info("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        logger.info("Creating H265 Encoder")
    if not encoder:
        logger.error(" Unable to create encoder")
    encoder.set_property("bitrate", bitrate)
    encoder.set_property("gpu-id", 0)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        logger.info("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        logger.info("Creating H265 rtppay")
    if not rtppay:
        logger.error(" Unable to create rtppay")

    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        logger.error(" Unable to create udpsink")

    sink.set_property("host", "224.224.255.255")
    sink.set_property("port", updsink_port_num)
    sink.set_property("async", False)
    sink.set_property("sync", 1)
    sink.set_property("qos", 0)

    streammux.set_property("width", 1080)
    streammux.set_property("height", 1920)
    streammux.set_property("batch-size", number_sources)
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("gpu-id", 0)

    if ts_from_rtsp:
        streammux.set_property("attach-sys-ts", 0)

    pgie.set_property("config-file-path", "yolov8_pgie_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        logger.warning("WARNING: Overriding infer-config batch-size %d with number of sources %d",
            pgie_batch_size, number_sources
        )
        pgie.set_property("batch-size", number_sources)

    logger.info("Adding elements to Pipeline")
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(tiler)
    tiler.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        logger.error("Unable to get src pad")
    else:
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    # Start streaming
    rtsp_port_num = 8554

    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    logger.info(f"*** DeepStream: Launched RTSP Streaming at rtsp://localhost:{rtsp_port_num}/ds-test ***")

    # start play back and listen to events
    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except BaseException:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                        help="Path to input H264 elementry stream", nargs="+", default=["a"], required=True)
    parser.add_argument("-c", "--codec", default="H264",
                        help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264', 'H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                        help="Set the encoding bitrate ", type=int)
    parser.add_argument("--rtsp-ts", action="store_true", default=True, dest='rtsp_ts',
                        help="Attach NTP timestamp from RTSP source",
                        )
    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global ts_from_rtsp
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    ts_from_rtsp = args.rtsp_ts
    return stream_path


if __name__ == '__main__':
    stream_path = parse_args()
    sys.exit(main(stream_path))
