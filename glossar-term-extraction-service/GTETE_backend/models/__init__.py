# coding: utf-8

# flake8: noqa
from __future__ import absolute_import
import os
# import models into model package

dirname = os.path.dirname(__file__)

from GTETE_backend.models.software_requirement import SoftwareRequirement  # noqa: E501
from GTETE_backend.models.statistics_table import StatisticsTable  # noqa: E501

