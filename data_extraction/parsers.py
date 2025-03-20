"""
Parsers for ChatGPT and TypingMind conversation exports.

This module provides parsers for extracting structured data from ChatGPT and TypingMind
conversation exports, converting them into a standardized format for further processing.
"""

import json
import os
import re
import datetime
from typing import Dict, List, Any, Optional, Tuple, Iterator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseParser:
    """Base class for conversation parsers."""
    
    def __init__(self, source_type: str):
        """
        Initialize the parser.
        
        Args:
            source_type: The type of source (e.g., 'chatgpt', 'typingmind')
        """
        self.source_type = source_type
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a conversation file.
        
        Args:
            file_path: Path to the conversation file
            
        Returns:
            Dict containing parsed conversation data
        """
        raise NotImplementedError("Subclasses must implement parse_file method")
    
    def extract_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract messages from parsed data.
        
        Args:
            data: Parsed conversation data
            
        Returns:
            List of message dictionaries
        """
        raise NotImplementedError("Subclasses must implement extract_messages method")
    
    def standardize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a message to the standard format.
        
        Args:
            message: Message data
            
        Returns:
            Standardized message dictionary
        """
        raise NotImplementedError("Subclasses must implement standardize_message method")
    
    def get_conversation_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract conversation metadata.
        
        Args:
            data: Parsed conversation data
            
        Returns:
            Dictionary of metadata
        """
        raise NotImplementedError("Subclasses must implement get_conversation_metadata method")


class ChatGPTParser(BaseParser):
    """Parser for ChatGPT conversation exports."""
    
    def __init__(self):
        """Initialize the ChatGPT parser."""
        super().__init__("chatgpt")
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a ChatGPT conversation file.
        
        Args:
            file_path: Path to the ChatGPT JSON file
            
        Returns:
            Dict containing parsed conversation data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract filename without extension for title if not present
            if 'title' not in data:
                base_name = os.path.basename(file_path)
                # Remove 'ChatGPT-' prefix and '.json' suffix if present
                title = re.sub(r'^ChatGPT-', '', base_name)
                title = re.sub(r'\.json$', '', title)
                # Replace underscores with spaces
                title = title.replace('_', ' ')
                data['title'] = title
                
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing ChatGPT file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing ChatGPT file {file_path}: {e}")
            raise
    
    def extract_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract messages from parsed ChatGPT data.
        
        Args:
            data: Parsed ChatGPT conversation data
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Handle different ChatGPT export formats
        if 'mapping' in data:
            # New format with mapping structure
            nodes = data.get('mapping', {})
            current_node_id = data.get('current_node')
            
            # Build a parent-child relationship map
            parent_map = {}
            for node_id, node_data in nodes.items():
                if 'parent' in node_data and node_data['parent']:
                    parent_map[node_id] = node_data['parent']
            
            # Reconstruct conversation by following parent links
            ordered_nodes = []
            node_id = current_node_id
            
            # First, go back to the root
            visited = set()
            while node_id and node_id not in visited:
                visited.add(node_id)
                ordered_nodes.insert(0, node_id)
                node_id = parent_map.get(node_id)
            
            # Extract messages from ordered nodes
            for node_id in ordered_nodes:
                if node_id in nodes:
                    node = nodes[node_id]
                    if 'message' in node and node['message']:
                        message = node['message']
                        if message.get('content') and message.get('author'):
                            messages.append(message)
        
        elif 'conversations' in data:
            # Older format with conversations array
            for conversation in data.get('conversations', []):
                messages.extend(conversation.get('messages', []))
        
        else:
            # Simplest format with direct messages array
            messages = data.get('messages', [])
        
        return messages
    
    def standardize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a ChatGPT message to the standard format.
        
        Args:
            message: ChatGPT message data
            
        Returns:
            Standardized message dictionary
        """
        # Extract content based on different possible structures
        content = ""
        if isinstance(message.get('content'), str):
            content = message['content']
        elif isinstance(message.get('content'), dict) and 'parts' in message['content']:
            parts = message['content']['parts']
            if parts and isinstance(parts, list):
                content = ''.join(str(part) for part in parts if isinstance(part, (str, int, float)))
        elif isinstance(message.get('content'), list):
            # Handle content as a list of content blocks
            for block in message['content']:
                if isinstance(block, dict) and 'type' in block and 'text' in block:
                    if block['type'] == 'text':
                        content += block['text'] + "\n"
                elif isinstance(block, str):
                    content += block + "\n"
        
        # Extract role
        role = ""
        if 'author' in message and isinstance(message['author'], dict):
            role = message['author'].get('role', '')
        else:
            role = message.get('role', '')
        
        # Extract timestamp
        timestamp = None
        if 'create_time' in message:
            timestamp = message['create_time']
        elif 'timestamp' in message:
            timestamp = message['timestamp']
        
        # Convert timestamp to ISO format if it's a number
        if timestamp and isinstance(timestamp, (int, float)):
            timestamp = datetime.datetime.fromtimestamp(timestamp).isoformat()
        
        # Extract message ID
        message_id = message.get('id', '')
        
        return {
            'id': message_id,
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'source': 'chatgpt',
            'metadata': {
                'model': message.get('model', ''),
                'original_message': message
            }
        }
    
    def get_conversation_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract conversation metadata from ChatGPT data.
        
        Args:
            data: Parsed ChatGPT conversation data
            
        Returns:
            Dictionary of metadata
        """
        # Extract conversation ID
        conversation_id = data.get('conversation_id', '')
        if not conversation_id and 'id' in data:
            conversation_id = data['id']
        
        # Extract title
        title = data.get('title', '')
        
        # Extract create time
        create_time = None
        if 'create_time' in data:
            create_time = data['create_time']
        elif 'createdAt' in data:
            create_time = data['createdAt']
        
        # Convert create_time to ISO format if it's a number
        if create_time and isinstance(create_time, (int, float)):
            create_time = datetime.datetime.fromtimestamp(create_time).isoformat()
        elif create_time and isinstance(create_time, str):
            # Try to parse string timestamp
            try:
                dt = datetime.datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                create_time = dt.isoformat()
            except ValueError:
                pass
        
        # Extract model information
        model = data.get('model', '')
        if not model:
            model = data.get('model_slug', '')
        
        return {
            'conversation_id': conversation_id,
            'title': title,
            'create_time': create_time,
            'model': model,
            'source': 'chatgpt'
        }


class TypingMindParser(BaseParser):
    """Parser for TypingMind conversation exports."""
    
    def __init__(self):
        """Initialize the TypingMind parser."""
        super().__init__("typingmind")
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a TypingMind conversation file.
        
        Args:
            file_path: Path to the TypingMind JSON file
            
        Returns:
            Dict containing parsed conversation data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract filename without extension for title if not present
            if 'title' not in data:
                base_name = os.path.basename(file_path)
                # Remove date prefix if present (format: YYYYMMDD_HHMMSS_)
                title = re.sub(r'^\d{8}_\d{6}_', '', base_name)
                # Remove '_export.json' suffix if present
                title = re.sub(r'_export\.json$', '', title)
                # Replace underscores with spaces
                title = title.replace('_', ' ')
                data['title'] = title
                
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing TypingMind file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing TypingMind file {file_path}: {e}")
            raise
    
    def extract_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract messages from parsed TypingMind data.
        
        Args:
            data: Parsed TypingMind conversation data
            
        Returns:
            List of message dictionaries
        """
        # TypingMind exports typically have a direct messages array
        return data.get('messages', [])
    
    def standardize_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a TypingMind message to the standard format.
        
        Args:
            message: TypingMind message data
            
        Returns:
            Standardized message dictionary
        """
        # Extract content based on different possible structures
        content = ""
        if isinstance(message.get('content'), str):
            content = message['content']
        elif isinstance(message.get('content'), list):
            # Handle content as a list of content blocks
            for block in message['content']:
                if isinstance(block, dict) and 'type' in block:
                    if block['type'] == 'text':
                        content += block.get('text', '') + "\n"
                    elif block['type'] == 'thinking':
                        # Skip thinking blocks or handle separately
                        pass
                elif isinstance(block, str):
                    content += block + "\n"
        
        # Extract role
        role = message.get('role', '')
        
        # Extract timestamp
        timestamp = None
        if 'createdAt' in message:
            timestamp = message['createdAt']
        
        # Convert timestamp to ISO format if needed
        if timestamp and isinstance(timestamp, str):
            # Try to parse string timestamp
            try:
                dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.isoformat()
            except ValueError:
                pass
        
        # Extract message ID
        message_id = message.get('id', '')
        
        # Extract attachments if present
        attachments = []
        if 'attachments' in message and isinstance(message['attachments'], list):
            for attachment in message['attachments']:
                if isinstance(attachment, dict):
                    attachments.append(attachment)
        
        return {
            'id': message_id,
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'source': 'typingmind',
            'metadata': {
                'model': message.get('model', ''),
                'attachments': attachments,
                'usage': message.get('usage', {}),
                'original_message': message
            }
        }
    
    def get_conversation_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract conversation metadata from TypingMind data.
        
        Args:
            data: Parsed TypingMind conversation data
            
        Returns:
            Dictionary of metadata
        """
        # Extract conversation ID
        conversation_id = data.get('id', '')
        
        # Extract title
        title = data.get('title', '')
        
        # Extract create time
        create_time = data.get('createdAt', '')
        
        # Convert create_time to ISO format if needed
        if create_time and isinstance(create_time, str):
            # Try to parse string timestamp
            try:
                dt = datetime.datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                create_time = dt.isoformat()
            except ValueError:
                pass
        
        # Extract model information from the first assistant message
        model = ''
        for message in data.get('messages', []):
            if message.get('role') == 'assistant' and 'model' in message:
                model = message['model']
                break
        
        return {
            'conversation_id': conversation_id,
            'title': title,
            'create_time': create_time,
            'model': model,
            'source': 'typingmind'
        }


class ConversationProcessor:
    """Process conversations from different sources into a standardized format."""
    
    def __init__(self):
        """Initialize the conversation processor."""
        self.parsers = {
            'chatgpt': ChatGPTParser(),
            'typingmind': TypingMindParser()
        }
    
    def detect_source_type(self, file_path: str) -> str:
        """
        Detect the source type based on the file path or content.
        
        Args:
            file_path: Path to the conversation file
            
        Returns:
            Source type string ('chatgpt' or 'typingmind')
        """
        base_name = os.path.basename(file_path).lower()
        
        if 'chatgpt' in base_name:
            return 'chatgpt'
        elif 'typingmind' in base_name:
            return 'typingmind'
        
        # If not determined by filename, try to peek at the content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_chunk = f.read(1000)  # Read first 1000 characters
                
            if '"mapping":' in first_chunk or '"conversations":' in first_chunk:
                return 'chatgpt'
            elif '"messages":' in first_chunk and ('"claude"' in first_chunk or '"gpt"' in first_chunk):
                return 'typingmind'
        except Exception:
            pass
        
        # Default to ChatGPT if unable to determine
        logger.warning(f"Unable to determine source type for {file_path}, defaulting to ChatGPT")
        return 'chatgpt'
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a conversation file into standardized format.
        
        Args:
            file_path: Path to the conversation file
            
        Returns:
            Dict containing standardized conversation data
        """
        source_type = self.detect_source_type(file_path)
        parser = self.parsers[source_type]
        
        # Parse the file
        data = parser.parse_file(file_path)
        
        # Extract metadata
        metadata = parser.get_conversation_metadata(data)
        
        # Extract and standardize messages
        raw_messages = parser.extract_messages(data)
        messages = [parser.standardize_message(msg) for msg in raw_messages]
        
        return {
            'metadata': metadata,
            'messages': messages,
            'source_file': file_path
        }
    
    def process_directory(self, directory_path: str) -> Iterator[Dict[str, Any]]:
        """
        Process all conversation files in a directory.
        
        Args:
            directory_path: Path to directory containing conversation files
            
        Yields:
            Dict containing standardized conversation data for each file
        """
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                try:
                    yield self.process_file(file_path)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
    
    def save_processed_conversation(self, conversation: Dict[str, Any], output_dir: str) -> str:
        """
        Save a processed conversation to a file.
        
        Args:
            conversation: Processed conversation data
            output_dir: Directory to save the file
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename from metadata
        metadata = conversation['metadata']
        source = metadata['source']
        conv_id = metadata.get('conversation_id', '')
        
        if not conv_id:
            # Generate a unique ID if conversation_id is not available
            conv_id = f"{source}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Clean the title for use in filename
        title = metadata.get('title', '')
        if title:
            # Remove special characters and limit length
            clean_title = re.sub(r'[^\w\s-]', '', title).strip()
            clean_title = re.sub(r'[-\s]+', '_', clean_title)
            clean_title = clean_title[:50]  # Limit length
            filename = f"{source}_{clean_title}_{conv_id}.json"
        else:
            filename = f"{source}_{conv_id}.json"
        
        # Save to file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        return output_path


# Example usage
if __name__ == "__main__":
    # Process a single file
    processor = ConversationProcessor()
    
    # Example paths
    sample_dir = "/home/ubuntu/llm_memory_system/data_extraction/sample_data"
    output_dir = "/home/ubuntu/llm_memory_system/data_extraction/processed_data"
    
    # Process all files in the sample directory
    os.makedirs(output_dir, exist_ok=True)
    
    for conversation in processor.process_directory(sample_dir):
        output_path = processor.save_processed_conversation(conversation, output_dir)
        print(f"Processed and saved: {output_path}")
