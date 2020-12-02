"""
This is a class for some helper methods to make logging more nice.
"""

import logging


def print_dict(dictionary:dict, header: str):
    """
    Method to print a dict in nice.
    :param dictionary:
    :param header:
    :return:
    """
    logging.info(f"\n ---{header.ljust(100,'-')}" + \
                 "\n" + '\n'.join('{}\t{}'.format(k.ljust(30).upper(), v) for k, v in dictionary.items()) + \
                 f"\n ---{''.ljust(100,'-')}")


def print_big_log(text:str, log_level:str="INFO")->None:
    """
    A method to log headlines.
    :param text: Text to print
    :return:
    """
    _level =logging.getLevelName(log_level.upper())
    logging.log(_level, "#########################################")
    logging.log(_level, f"### {text}")
    logging.log(_level, "########################################")
