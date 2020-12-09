"""
This is a class for some helper methods to make logging more nice.
"""

import logging
import sys

COLORS = {
    "RESET": '\x1B[0m',
    "RED": '\x1B[31m',
    "YELLOW": "\x1B[33m",
    "BRGREEN": '\x1B[01;32m',
    "DARKGREEN": '\x1B[0;32m',
    "BOLD": '\x1b[5m',

    "BRIGHTGREY": "\x1b[1;37m",
    "DARKGREY": "\x1b[0;37m",

    "DARKBLUE": '\x1B[0;36m',
    "BRIGHTBLUE": '\x1B[1;36m'
}


def log_step(_episode, step, reward, action):
    logging.debug(
        f"--EPISODE {(str(_episode + 1).ljust(2))}.{str(step).ljust(4)} | {colored_log_text(f'rew: {reward:.4f}', 'DARKGREEN')} | action: {action} ")


def log_episode(_episode, step, reward, p_loss, q_loss, a_loss,time, level="DEBUG"):
    # p_loss = sum(p_loss) / len(p_loss) if len(p_loss) != 0 else -1
    # q_loss = sum(q_loss) / len(q_loss) if len(q_loss) != 0 else -1

    level = logging.getLevelName(level)
    logging.log(level,
                f"EPISODE {str(_episode + 1).ljust(4)} | Reward {reward:.4f} | P-Loss {p_loss:.4f} | Q-Loss {q_loss:.4f} | a-Loss {a_loss:.4f} | time {time:0.2f}s"
                )


def setup_logging(args):
    # Set the logging format
    format = '{asctime} [{filename}:{lineno}] {levelname:8} {message}'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Setup coloring
    h = ColouredHandler()
    h.formatter = ColouredFormatter(format, date_format, '{')

    file_handler = logging.FileHandler(args.get('log_file'),
                                       mode='w+')
    file_handler.formatter = ColouredFormatter(format, date_format, '{')

    # Setup the logging environment
    level = logging.getLevelName(args.get('log_level'))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    logging.basicConfig(datefmt=date_format,
                        level=int(level),
                        handlers=[file_handler, h]
                        )

    logger = logging.getLogger(__name__)
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)


def colored_log_text(txt, color):
    return COLORS.get(color) + txt + COLORS.get('RESET')


def print_dict(dictionary: dict, header: str):
    """
    Method to print a dict in nice.
    :param dictionary:
    :param header:
    :return:
    """
    logging.info(colored_log_text(f"\n\n ---{header.ljust(70, '-')}", "DARKBLUE") + \
                 '\n' + '\n'.join((colored_log_text('{}', 'DARKBLUE') + '\t' +
                                   (colored_log_text('{}', 'BRIGHTBLUE'))).format(k.ljust(20).upper(), v) for k, v in
                                  dictionary.items()) + \
                 colored_log_text(f"\n ---{''.ljust(70, '-')}\n\n", "DARKBLUE"))


def print_step_log(text: str, log_level: str = "INFO") -> None:
    """
    Helper Method to plot a step.
    :param text:
    :param log_level:
    :return:
    """
    _level = logging.getLevelName(log_level.upper())
    logging.log(_level, colored_log_text(f"{text} ...", "DARKBLUE"))
    logging.log(_level, colored_log_text("-------------------------------------------", "DARKBLUE"))


def print_big_log(text: str, log_level: str = "INFO") -> None:
    """
    A method to log headlines.
    :param text: Text to print
    :return:
    """
    _level = logging.getLevelName(log_level.upper())
    logging.log(_level, colored_log_text("#########################################", "DARKBLUE"))
    logging.log(_level, colored_log_text(f"### {text}", "DARKBLUE"))
    logging.log(_level, colored_log_text("########################################", "DARKBLUE"))


# Class to make colored output.
class ColouredFormatter(logging.Formatter):
    def format(self, record, colour=False):
        message = super().format(record)

        if not colour:
            return message

        level_no = record.levelno
        if level_no >= logging.CRITICAL:
            colour = COLORS.get("RED")
        elif level_no >= logging.ERROR:
            colour = COLORS.get("RED")
        elif level_no >= logging.WARNING:
            colour = COLORS.get("YELLOW")
        elif level_no >= logging.INFO:
            colour = COLORS.get("BRIGHTGREY")
        elif level_no >= logging.DEBUG:
            colour = COLORS.get("DARKGREY")
        else:
            colour = COLORS.get("RESET")

        message = colour + message + COLORS.get("RESET")

        return message


class ColouredHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)

    def format(self, record, colour=False):
        if not isinstance(self.formatter, ColouredFormatter):
            self.formatter = ColouredFormatter()

        return self.formatter.format(record, colour)

    def emit(self, record):
        stream = self.stream
        try:
            msg = self.format(record, stream.isatty())
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)
