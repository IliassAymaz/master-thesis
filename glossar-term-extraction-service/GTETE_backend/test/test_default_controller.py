# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from GTETE_backend.models.software_requirement import SoftwareRequirement  # noqa: E501
from GTETE_backend.models.statistics_table import StatisticsTable  # noqa: E501
from GTETE_backend.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_get_statistics(self):
        """Test case for get_statistics

        Return statistic table with absolute and relative term frequencies for all verbs
        """
        requirements_list = [SoftwareRequirement()]
        response = self.client.open(
            '/create_statistics',
            method='POST',
            data=json.dumps(requirements_list),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_ping_get(self):
        """Test case for ping_get

        
        """
        response = self.client.open(
            '/ping',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_version_get(self):
        """Test case for version_get

        
        """
        response = self.client.open(
            '/version',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
