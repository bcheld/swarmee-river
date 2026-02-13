#!/usr/bin/env python3
"""
Unit tests for the welcome tool
"""

from unittest import mock

from tools.welcome import DEFAULT_WELCOME_TEXT, welcome


class TestWelcomeTool:
    """Test cases for the welcome tool"""

    def test_view_welcome_default(self):
        """Test viewing welcome text with default content"""
        with mock.patch("pathlib.Path.exists", return_value=False), mock.patch("pathlib.Path.read_text") as mock_read:
            tool = {"toolUseId": "test-id", "input": {"action": "view"}}

            result = welcome(tool)
            assert result["status"] == "success"
            assert DEFAULT_WELCOME_TEXT in result["content"][0]["text"]
            assert "First things to try" in result["content"][0]["text"]
            mock_read.assert_not_called()

    def test_view_welcome_custom(self):
        """Test viewing welcome text with custom content"""
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "pathlib.Path.read_text", return_value="Custom welcome text"
        ) as mock_read:
            tool = {"toolUseId": "test-id", "input": {"action": "view"}}

            result = welcome(tool)

            assert result["status"] == "success"
            assert result["content"][0]["text"] == "Custom welcome text"
            mock_read.assert_called_once()

    def test_edit_welcome(self):
        """Test editing welcome text"""
        with mock.patch("pathlib.Path.write_text") as mock_write_text:
            tool = {"toolUseId": "test-id", "input": {"action": "edit", "content": "New welcome text"}}

            result = welcome(tool)

            assert result["status"] == "success"
            assert "updated successfully" in result["content"][0]["text"]
            assert mock_write_text.call_count == 1
            assert mock_write_text.call_args.args[0] == "New welcome text"
            assert mock_write_text.call_args.kwargs == {"encoding": "utf-8"}

    def test_edit_welcome_missing_content(self):
        """Test editing welcome text with missing content"""
        tool = {
            "toolUseId": "test-id",
            "input": {
                "action": "edit"
                # Missing content parameter
            },
        }

        result = welcome(tool)

        # Check that an error was returned
        assert result["status"] == "error"
        assert "content is required" in result["content"][0]["text"]

    def test_unknown_action(self):
        """Test welcome with unknown action"""
        tool = {"toolUseId": "test-id", "input": {"action": "unknown"}}

        result = welcome(tool)

        # Check that an error was returned
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_file_operation_error(self):
        """Test error during file operations"""
        with mock.patch("pathlib.Path.exists", return_value=True), mock.patch(
            "pathlib.Path.read_text", side_effect=PermissionError("Permission denied")
        ):
            tool = {"toolUseId": "test-id", "input": {"action": "view"}}

            result = welcome(tool)

            assert result["status"] == "error"
            assert "Error" in result["content"][0]["text"]
            assert "Permission denied" in result["content"][0]["text"]
