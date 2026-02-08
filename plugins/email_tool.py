#!/usr/bin/env python3
"""
Email Tool Plugin for SAM Agent
Sends email notifications using Amazon SES or SMTP
"""

import json
import logging
from typing import Dict, Any, Optional, List

# Try to import boto3 for SES support
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Try to import smtplib for SMTP support
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

from sam_agent import SAMPlugin, ToolCategory

logger = logging.getLogger("SAM.EmailTool")


class EmailToolPlugin(SAMPlugin):
    """Email notification plugin for SAM Agent"""

    def __init__(self):
        super().__init__(
            name="Email Notifications",
            version="1.0.0",
            description="Send email notifications via Amazon SES or SMTP"
        )
        self.config = None
        self.ses_client = None

    def on_load(self, agent):
        """Initialize email configuration from agent config"""
        if hasattr(agent.config, 'email'):
            self.config = agent.config.email
            
            # Initialize SES client if using AWS SES
            if self.config.enabled and self.config.provider == "ses" and HAS_BOTO3:
                try:
                    self.ses_client = boto3.client(
                        'ses',
                        region_name=self.config.aws_region,
                        aws_access_key_id=self.config.aws_access_key_id,
                        aws_secret_access_key=self.config.aws_secret_access_key
                    )
                    logger.info(f"✉️ Email tools connected via Amazon SES ({self.config.aws_region})")
                except Exception as e:
                    logger.error(f"Failed to initialize SES client: {e}")
            elif self.config.enabled and self.config.provider == "smtp":
                logger.info(f"✉️ Email tools connected via SMTP ({self.config.smtp_host})")
        else:
            logger.warning("Email config not found - tool disabled")
            # Create a disabled default config
            class DefaultConfig:
                enabled = False
            self.config = DefaultConfig()

    def register_tools(self, agent):
        """Register email tools with the agent"""
        if not self.config or not self.config.enabled:
            return
        
        agent.register_local_tool(
            self.send_email,
            category=ToolCategory.COMMUNICATION,
            requires_approval=False  # Safe operation
        )
        
        agent.register_local_tool(
            self.check_dependencies,
            category=ToolCategory.UTILITY,
            requires_approval=False
        )

    def check_dependencies(self, query: str = "") -> str:
        """
        Check email tool configuration and dependencies status
        
        Args:
            query: Optional query (ignored, for compatibility)
            
        Returns:
            Configuration and dependency status
        """
        status = "📧 Email Tool Status\n"
        status += "=" * 50 + "\n\n"
        
        if not self.config:
            return status + "⚠️ Configuration not loaded yet\n"
        
        status += f"Enabled: {'✅' if self.config.enabled else '❌'}\n"
        status += f"Provider: {self.config.provider}\n"
        
        if self.config.provider == "ses":
            status += f"\n🔧 Amazon SES Configuration:\n"
            status += f"  Region: {self.config.aws_region}\n"
            status += f"  Access Key: {'✅ Set' if self.config.aws_access_key_id else '❌ Not set'}\n"
            status += f"  Secret Key: {'✅ Set' if self.config.aws_secret_access_key else '❌ Not set'}\n"
            status += f"  boto3: {'✅ Installed' if HAS_BOTO3 else '❌ Not installed (pip install boto3)'}\n"
            status += f"  SES Client: {'✅ Initialized' if self.ses_client else '❌ Not initialized'}\n"
        elif self.config.provider == "smtp":
            status += f"\n🔧 SMTP Configuration:\n"
            status += f"  Host: {self.config.smtp_host or '❌ Not set'}\n"
            status += f"  Port: {self.config.smtp_port}\n"
            status += f"  Username: {'✅ Set' if self.config.smtp_username else '❌ Not set'}\n"
            status += f"  Password: {'✅ Set' if self.config.smtp_password else '❌ Not set'}\n"
            status += f"  TLS: {'✅' if self.config.smtp_use_tls else '❌'}\n"
        
        status += f"\n📬 Email Addresses:\n"
        status += f"  From: {self.config.from_email or '❌ Not set'}\n"
        status += f"  To: {self.config.to_email or '❌ Not set'}\n"
        if self.config.reply_to:
            status += f"  Reply-To: {self.config.reply_to}\n"
        
        status += f"\n🔔 Notifications:\n"
        status += f"  On Completion: {'✅' if self.config.notify_on_completion else '❌'}\n"
        status += f"  On Error: {'✅' if self.config.notify_on_error else '❌'}\n"
        status += f"  On Approval: {'✅' if self.config.notify_on_approval else '❌'}\n"
        
        return status

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        is_html: bool = False,
        attachment_data: Optional[str] = None
    ) -> str:
        """
        Send an email to any recipient via SMTP or Amazon SES
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            body: Email body content
            is_html: Whether body is HTML formatted (default: False)
            attachment_data: Optional text data to attach as file
            
        Returns:
            Status message string
            
        Examples:
            send_email("user@example.com", "Task Complete", "Your task has finished")
            send_email("colleague@company.com", "Meeting Reminder", "Don't forget tomorrow")
            send_email("admin@example.com", "Error Alert", "<h1>Error</h1>", is_html=True)
        """
        import datetime
        exec_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        logger.info(f"📧 [{exec_time}] send_email() ACTUALLY CALLED - to: {to_email}, subject: {subject}")
        
        if not self.config or not self.config.enabled:
            return "❌ Email tool is not enabled in configuration"

        # Get email addresses from config
        from_email = self.config.from_email
        recipient = to_email
        reply_to = self.config.reply_to or from_email

        if not from_email or not recipient:
            return "❌ From and To email addresses must be configured in config.json"

        try:
            if self.config.provider == "ses":
                result = self._send_via_ses(
                    from_email, recipient, reply_to,
                    subject, body, is_html, attachment_data
                )
            elif self.config.provider == "smtp":
                result = self._send_via_smtp(
                    from_email, recipient, reply_to,
                    subject, body, is_html, attachment_data
                )
            else:
                return f"❌ Unsupported email provider: {self.config.provider}"

            if result["success"]:
                return f"✅ {result['message']}"
            else:
                return f"❌ {result['error']}"

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return f"❌ Failed to send email: {str(e)}"

    def _send_via_ses(
        self,
        from_email: str,
        to_email: str,
        reply_to: str,
        subject: str,
        body: str,
        is_html: bool,
        attachment_data: Optional[str]
    ) -> Dict[str, Any]:
        """Send email via Amazon SES"""
        if not HAS_BOTO3:
            return {
                "success": False,
                "error": "boto3 library not installed. Install with: pip install boto3"
            }

        if not self.ses_client:
            return {
                "success": False,
                "error": "SES client not initialized"
            }

        try:
            # Build email message
            message = {
                'Subject': {'Data': subject, 'Charset': 'UTF-8'},
                'Body': {}
            }

            if is_html:
                message['Body']['Html'] = {'Data': body, 'Charset': 'UTF-8'}
            else:
                message['Body']['Text'] = {'Data': body, 'Charset': 'UTF-8'}

            # Send email
            response = self.ses_client.send_email(
                Source=from_email,
                Destination={'ToAddresses': [to_email]},
                Message=message,
                ReplyToAddresses=[reply_to] if reply_to else []
            )

            logger.info(f"✉️ Email sent successfully via SES: {response['MessageId']}")
            return {
                "success": True,
                "message": f"Email sent successfully to {to_email}",
                "message_id": response['MessageId']
            }

        except ClientError as e:
            error_msg = e.response['Error']['Message']
            logger.error(f"SES ClientError: {error_msg}")
            return {
                "success": False,
                "error": f"SES error: {error_msg}"
            }
        except Exception as e:
            logger.error(f"Failed to send email via SES: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _send_via_smtp(
        self,
        from_email: str,
        to_email: str,
        reply_to: str,
        subject: str,
        body: str,
        is_html: bool,
        attachment_data: Optional[str]
    ) -> Dict[str, Any]:
        """Send email via SMTP"""
        smtp_host = self.config.smtp_host or ""
        smtp_port = self.config.smtp_port
        smtp_username = self.config.smtp_username or ""
        smtp_password = self.config.smtp_password or ""
        smtp_use_tls = self.config.smtp_use_tls

        if not smtp_host:
            return {
                "success": False,
                "error": "SMTP host not configured"
            }

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = formataddr(("SAM Agent", from_email))
            msg['To'] = to_email
            if reply_to:
                msg['Reply-To'] = reply_to

            # Attach body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            # Attach optional data
            if attachment_data:
                attachment = MIMEText(attachment_data, 'plain')
                attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename='data.txt'
                )
                msg.attach(attachment)

            # Send email
            import datetime
            exec_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            with smtplib.SMTP(smtp_host, smtp_port, timeout=self.config.timeout) as server:
                logger.info(f"📧 [{exec_time}] Connecting to SMTP server {smtp_host}:{smtp_port}...")
                if smtp_use_tls:
                    server.starttls()
                    logger.info(f"📧 [{exec_time}] TLS started")
                
                if smtp_username and smtp_password:
                    logger.info(f"📧 [{exec_time}] Authenticating as {smtp_username}...")
                    server.login(smtp_username, smtp_password)
                    logger.info(f"📧 [{exec_time}] Authentication successful")
                
                logger.info(f"📧 [{exec_time}] Sending message to {to_email}...")
                server.send_message(msg)
                logger.info(f"📧 [{exec_time}] Message sent successfully!")

            logger.info(f"✉️ Email sent successfully via SMTP to {to_email}")
            return {
                "success": True,
                "message": f"Email sent successfully to {to_email}"
            }

        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return {
                "success": False,
                "error": f"SMTP error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def create_plugin():
    """Factory function to create the plugin instance"""
    return EmailToolPlugin()
