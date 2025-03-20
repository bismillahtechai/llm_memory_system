# Dynamic Context Management System Design

## Overview

This document outlines the design for the Dynamic Context Management system, which is responsible for determining what information to store in memory, when to retrieve it, and how to optimize context inclusion in prompts to minimize token usage while maximizing relevance.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Dynamic Context Management System                   │
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Query         │    │ Memory        │    │ Context                │    │
│  │ Analyzer      │───▶│ Retrieval     │───▶│ Optimization          │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
│                                                        │                 │
│                                                        ▼                 │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Memory        │◀───│ Response      │◀───│ Prompt                │    │
│  │ Update        │    │ Analyzer      │    │ Assembly              │    │
│  └───────────────┘    └───────────────┘    └───────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Query Analyzer

#### Functionality:
- Analyzes user queries to determine information needs
- Classifies query types (factual, personal, procedural, etc.)
- Identifies entities and topics mentioned in the query
- Determines which memory systems to query (RAG, Mem0, or both)

#### Implementation:
```python
class QueryAnalyzer:
    def __init__(self, nlp_model=None):
        """
        Initialize the Query Analyzer.
        
        Args:
            nlp_model: Optional NLP model for advanced analysis
        """
        self.nlp_model = nlp_model
    
    def analyze(self, query, conversation_history=None):
        """
        Analyze a user query to determine memory retrieval strategy.
        
        Args:
            query: The user's query text
            conversation_history: Optional recent conversation history
            
        Returns:
            dict: Analysis results including query type, entities, and retrieval strategy
        """
        # Basic analysis
        query_length = len(query.split())
        has_question = '?' in query
        
        # Extract entities and keywords
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        # Determine query type
        query_type = self._classify_query_type(query, has_question)
        
        # Determine memory systems to query
        memory_systems = self._determine_memory_systems(query_type, entities, keywords)
        
        # Create retrieval parameters
        retrieval_params = {
            'rag_params': self._create_rag_params(query_type, entities, keywords),
            'mem0_params': self._create_mem0_params(query_type, entities, keywords)
        }
        
        return {
            'query_type': query_type,
            'entities': entities,
            'keywords': keywords,
            'memory_systems': memory_systems,
            'retrieval_params': retrieval_params
        }
    
    def _extract_entities(self, text):
        """Extract named entities from text"""
        if self.nlp_model:
            # Use NLP model for entity extraction
            return [ent.text for ent in self.nlp_model(text).ents]
        else:
            # Simple keyword-based entity extraction
            # This is a placeholder for actual implementation
            return []
    
    def _extract_keywords(self, text):
        """Extract important keywords from text"""
        # Simple implementation - remove stopwords and get unique terms
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        words = [word.lower() for word in re.findall(r'\b\w+\b', text)]
        return [word for word in words if word not in stopwords]
    
    def _classify_query_type(self, query, has_question):
        """Classify the type of query"""
        # Simple rule-based classification
        if has_question:
            if any(word in query.lower() for word in ['who', 'what', 'where', 'when', 'why', 'how']):
                return 'factual'
            elif any(word in query.lower() for word in ['you', 'your', 'we', 'our', 'remember']):
                return 'personal'
            else:
                return 'general'
        else:
            if any(word in query.lower() for word in ['please', 'could', 'would', 'can']):
                return 'instruction'
            else:
                return 'statement'
    
    def _determine_memory_systems(self, query_type, entities, keywords):
        """Determine which memory systems to query"""
        systems = []
        
        # Add RAG for factual queries
        if query_type in ['factual', 'general']:
            systems.append('rag')
        
        # Add Mem0 for personal queries
        if query_type in ['personal', 'instruction']:
            systems.append('mem0')
        
        # If no specific system is indicated, query both
        if not systems:
            systems = ['rag', 'mem0']
        
        return systems
    
    def _create_rag_params(self, query_type, entities, keywords):
        """Create parameters for RAG retrieval"""
        return {
            'top_k': 5 if query_type == 'factual' else 3,
            'filter_criteria': {
                'entities': entities
            },
            'reranking': query_type == 'factual'
        }
    
    def _create_mem0_params(self, query_type, entities, keywords):
        """Create parameters for Mem0 retrieval"""
        return {
            'preference_weight': 0.8 if query_type == 'personal' else 0.4,
            'recency_weight': 0.6,
            'entities': entities
        }
```

### 2. Memory Retrieval

#### Functionality:
- Queries appropriate memory systems based on analysis
- Retrieves relevant information from RAG and/or Mem0
- Applies filtering and ranking to results
- Handles retrieval errors and fallbacks

#### Implementation:
```python
class MemoryRetriever:
    def __init__(self, rag_system, mem0_system):
        """
        Initialize the Memory Retriever.
        
        Args:
            rag_system: RAG component instance
            mem0_system: Mem0 component instance
        """
        self.rag_system = rag_system
        self.mem0_system = mem0_system
    
    def retrieve(self, query, analysis):
        """
        Retrieve relevant memories based on query analysis.
        
        Args:
            query: The user's query text
            analysis: Results from QueryAnalyzer
            
        Returns:
            dict: Retrieved memories from different systems
        """
        results = {
            'rag_results': [],
            'mem0_results': [],
            'combined_results': []
        }
        
        # Query RAG system if needed
        if 'rag' in analysis['memory_systems']:
            try:
                rag_results = self.rag_system.query(
                    query,
                    top_k=analysis['retrieval_params']['rag_params']['top_k'],
                    filter_criteria=analysis['retrieval_params']['rag_params']['filter_criteria'],
                    reranking=analysis['retrieval_params']['rag_params']['reranking']
                )
                results['rag_results'] = rag_results
            except Exception as e:
                print(f"RAG retrieval error: {str(e)}")
        
        # Query Mem0 system if needed
        if 'mem0' in analysis['memory_systems']:
            try:
                mem0_results = self.mem0_system.query(
                    query,
                    preference_weight=analysis['retrieval_params']['mem0_params']['preference_weight'],
                    recency_weight=analysis['retrieval_params']['mem0_params']['recency_weight'],
                    entities=analysis['retrieval_params']['mem0_params']['entities']
                )
                results['mem0_results'] = mem0_results
            except Exception as e:
                print(f"Mem0 retrieval error: {str(e)}")
        
        # Combine and prioritize results
        results['combined_results'] = self._combine_results(
            results['rag_results'],
            results['mem0_results'],
            analysis['query_type']
        )
        
        return results
    
    def _combine_results(self, rag_results, mem0_results, query_type):
        """
        Combine and prioritize results from different memory systems.
        
        Args:
            rag_results: Results from RAG system
            mem0_results: Results from Mem0 system
            query_type: Type of query being processed
            
        Returns:
            list: Combined and prioritized results
        """
        combined = []
        
        # Determine priority based on query type
        if query_type in ['factual', 'general']:
            # Prioritize RAG for factual queries
            primary, secondary = rag_results, mem0_results
            primary_weight, secondary_weight = 0.7, 0.3
        elif query_type in ['personal', 'instruction']:
            # Prioritize Mem0 for personal queries
            primary, secondary = mem0_results, rag_results
            primary_weight, secondary_weight = 0.7, 0.3
        else:
            # Equal priority for other query types
            primary, secondary = rag_results, mem0_results
            primary_weight, secondary_weight = 0.5, 0.5
        
        # Add primary results first
        combined.extend([(item, primary_weight) for item in primary])
        
        # Add secondary results, avoiding duplicates
        primary_ids = {item['id'] for item in primary}
        for item in secondary:
            if item['id'] not in primary_ids:
                combined.append((item, secondary_weight))
        
        # Sort by weighted relevance score
        combined.sort(key=lambda x: x[0]['relevance'] * x[1], reverse=True)
        
        # Return just the items without weights
        return [item for item, _ in combined]
```

### 3. Context Optimization

#### Functionality:
- Optimizes retrieved context to fit within token limits
- Applies compression techniques to maximize information density
- Prioritizes content based on relevance and importance
- Balances different memory types in the final context

#### Implementation:
```python
class ContextOptimizer:
    def __init__(self, tokenizer, max_tokens=1000):
        """
        Initialize the Context Optimizer.
        
        Args:
            tokenizer: Tokenizer for counting tokens
            max_tokens: Maximum tokens allowed for context
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    
    def optimize(self, retrieved_memories, query_type):
        """
        Optimize retrieved memories to fit within token budget.
        
        Args:
            retrieved_memories: Results from MemoryRetriever
            query_type: Type of query being processed
            
        Returns:
            dict: Optimized context for inclusion in prompt
        """
        # Allocate token budget based on query type
        budget = self._allocate_budget(query_type)
        
        # Prepare memories for optimization
        rag_memories = retrieved_memories['rag_results']
        mem0_memories = retrieved_memories['mem0_results']
        
        # Optimize RAG memories
        optimized_rag = self._optimize_rag_memories(
            rag_memories, 
            budget['rag']
        )
        
        # Optimize Mem0 memories
        optimized_mem0 = self._optimize_mem0_memories(
            mem0_memories, 
            budget['mem0']
        )
        
        return {
            'rag_context': optimized_rag,
            'mem0_context': optimized_mem0,
            'token_usage': {
                'rag': self._count_tokens(optimized_rag),
                'mem0': self._count_tokens(optimized_mem0)
            }
        }
    
    def _allocate_budget(self, query_type):
        """Allocate token budget based on query type"""
        if query_type in ['factual', 'general']:
            # More tokens for RAG in factual queries
            rag_ratio = 0.7
        elif query_type in ['personal', 'instruction']:
            # More tokens for Mem0 in personal queries
            rag_ratio = 0.3
        else:
            # Equal distribution for other query types
            rag_ratio = 0.5
        
        rag_budget = int(self.max_tokens * rag_ratio)
        mem0_budget = self.max_tokens - rag_budget
        
        return {
            'rag': rag_budget,
            'mem0': mem0_budget
        }
    
    def _optimize_rag_memories(self, memories, budget):
        """Optimize RAG memories to fit within budget"""
        if not memories:
            return ""
        
        # Sort by relevance
        sorted_memories = sorted(memories, key=lambda x: x['relevance'], reverse=True)
        
        # Start with most relevant memories
        context = []
        current_tokens = 0
        
        for memory in sorted_memories:
            memory_text = memory['content']
            memory_tokens = self._count_tokens(memory_text)
            
            # If adding this memory exceeds budget, try compression
            if current_tokens + memory_tokens > budget:
                # Try to compress if it's a long memory
                if memory_tokens > 100:
                    compressed = self._compress_text(memory_text, budget - current_tokens)
                    compressed_tokens = self._count_tokens(compressed)
                    
                    if current_tokens + compressed_tokens <= budget:
                        context.append(compressed)
                        current_tokens += compressed_tokens
                
                # Skip if we can't fit it even with compression
                continue
            
            # Add memory if it fits
            context.append(memory_text)
            current_tokens += memory_tokens
            
            # Stop if we've reached the budget
            if current_tokens >= budget:
                break
        
        return "\n\n".join(context)
    
    def _optimize_mem0_memories(self, memories, budget):
        """Optimize Mem0 memories to fit within budget"""
        if not memories:
            return ""
        
        # For Mem0, we want to include a mix of preferences and facts
        preferences = [m for m in memories if m['type'] == 'preference']
        facts = [m for m in memories if m['type'] == 'fact']
        
        # Allocate budget between preferences and facts
        pref_budget = int(budget * 0.6)
        facts_budget = budget - pref_budget
        
        # Optimize preferences
        pref_context = self._optimize_mem0_section(preferences, pref_budget)
        
        # Optimize facts
        facts_context = self._optimize_mem0_section(facts, facts_budget)
        
        # Combine sections
        if pref_context and facts_context:
            return f"User preferences:\n{pref_context}\n\nRelevant facts:\n{facts_context}"
        elif pref_context:
            return f"User preferences:\n{pref_context}"
        elif facts_context:
            return f"Relevant facts:\n{facts_context}"
        else:
            return ""
    
    def _optimize_mem0_section(self, memories, budget):
        """Optimize a section of Mem0 memories"""
        if not memories:
            return ""
        
        # Sort by relevance
        sorted_memories = sorted(memories, key=lambda x: x['relevance'], reverse=True)
        
        # Start with most relevant memories
        context = []
        current_tokens = 0
        
        for memory in sorted_memories:
            memory_text = f"- {memory['content']}"
            memory_tokens = self._count_tokens(memory_text)
            
            if current_tokens + memory_tokens <= budget:
                context.append(memory_text)
                current_tokens += memory_tokens
            else:
                break
        
        return "\n".join(context)
    
    def _count_tokens(self, text):
        """Count tokens in text using the tokenizer"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))
    
    def _compress_text(self, text, target_tokens):
        """Compress text to fit within target token count"""
        # Simple compression: truncate and add ellipsis
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= target_tokens:
            return text
        
        # Leave room for ellipsis
        truncated_tokens = tokens[:target_tokens - 3]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        return truncated_text + "..."
```

### 4. Prompt Assembly

#### Functionality:
- Assembles the final prompt with optimized context
- Formats memory context for effective LLM utilization
- Includes appropriate system instructions
- Ensures the complete prompt fits within model context limits

#### Implementation:
```python
class PromptAssembler:
    def __init__(self, system_instruction_template, max_total_tokens=4000):
        """
        Initialize the Prompt Assembler.
        
        Args:
            system_instruction_template: Template for system instructions
            max_total_tokens: Maximum tokens for the entire prompt
        """
        self.system_instruction_template = system_instruction_template
        self.max_total_tokens = max_total_tokens
    
    def assemble(self, query, optimized_context, conversation_history, tokenizer):
        """
        Assemble the final prompt with optimized context.
        
        Args:
            query: The user's query text
            optimized_context: Results from ContextOptimizer
            conversation_history: Recent conversation history
            tokenizer: Tokenizer for counting tokens
            
        Returns:
            dict: Assembled prompt components
        """
        # Calculate token usage so far
        rag_tokens = optimized_context['token_usage']['rag']
        mem0_tokens = optimized_context['token_usage']['mem0']
        memory_tokens = rag_tokens + mem0_tokens
        
        # Format memory context
        memory_context = self._format_memory_context(
            optimized_context['rag_context'],
            optimized_context['mem0_context']
        )
        
        # Create system instructions with memory context
        system_instructions = self.system_instruction_template.format(
            memory_context=memory_context
        )
        system_tokens = len(tokenizer.encode(system_instructions))
        
        # Calculate remaining tokens for conversation history
        query_tokens = len(tokenizer.encode(query))
        available_for_history = self.max_total_tokens - system_tokens - query_tokens - 50  # 50 tokens buffer
        
        # Optimize conversation history to fit
        optimized_history = self._optimize_conversation_history(
            conversation_history,
            available_for_history,
            tokenizer
        )
        history_tokens = len(tokenizer.encode(json.dumps(optimized_history)))
        
        # Assemble final prompt components
        prompt = {
            'system': system_instructions,
            'messages': optimized_history + [{'role': 'user', 'content': query}],
            'token_usage': {
                'system': system_tokens,
                'history': history_tokens,
                'query': query_tokens,
                'memory': memory_tokens,
                'total': system_tokens + history_tokens + query_tokens
            }
        }
        
        return prompt
    
    def _format_memory_context(self, rag_context, mem0_context):
        """Format memory context for inclusion in system instructions"""
        sections = []
        
        if rag_context:
            sections.append(f"Relevant information from past conversations:\n{rag_context}")
        
        if mem0_context:
            sections.append(f"Personal context:\n{mem0_context}")
        
        if sections:
            return "\n\n".join(sections)
        else:
            return "No relevant context available."
    
    def _optimize_conversation_history(self, history, available_tokens, tokenizer):
        """Optimize conversation history to fit within available tokens"""
        if not history:
            return []
        
        # Start with most recent messages
        optimized = []
        current_tokens = 0
        
        # Process in reverse order (most recent first)
        for message in reversed(history):
            message_tokens = len(tokenizer.encode(json.dumps(message)))
            
            if current_tokens + message_tokens <= available_tokens:
                optimized.insert(0, message)  # Insert at beginning to maintain order
                current_tokens += message_tokens
            else:
                break
        
        return optimized
```

### 5. Response Analyzer

#### Functionality:
- Analyzes LLM responses to identify memory-worthy information
- Detects preferences, facts, and important insights
- Determines what should be stored in long-term memory
- Prepares extracted information for memory updates

#### Implementation:
```python
class ResponseAnalyzer:
    def __init__(self, nlp_model=None):
        """
        Initialize the Response Analyzer.
        
        Args:
            nlp_model: Optional NLP model for advanced analysis
        """
        self.nlp_model = nlp_model
    
    def analyze(self, query, response, conversation_history=None):
        """
        Analyze LLM response to identify memory-worthy information.
        
        Args:
            query: The user's query text
            response: The LLM's response text
            conversation_history: Optional recent conversation history
            
        Returns:
            dict: Analysis results including memory-worthy information
        """
        # Extract potential memory items
        memory_items = {
            'facts': self._extract_facts(query, response),
            'preferences': self._extract_preferences(query, response),
            'insights': self._extract_insights(query, response)
        }
        
        # Score items for memory-worthiness
        scored_items = self._score_memory_items(memory_items)
        
        # Filter items based on threshold
        memory_worthy = self._filter_memory_worthy(scored_items, threshold=0.6)
        
        return {
            'memory_worthy': memory_worthy,
            'all_items': scored_items
        }
    
    def _extract_facts(self, query, response):
        """Extract factual information from response"""
        facts = []
        
        # Simple rule-based extraction
        sentences = re.split(r'[.!?]', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Look for factual indicators
            if any(phrase in sentence.lower() for phrase in [
                'is a', 'are a', 'was a', 'were a',
                'is an', 'are an', 'was an', 'were an',
                'is the', 'are the', 'was the', 'were the',
                'has', 'have', 'had',
                'consists of', 'contains', 'includes'
            ]):
                facts.append({
                    'type': 'fact',
                    'content': sentence,
                    'source': 'response'
                })
        
        return facts
    
    def _extract_preferences(self, query, response):
        """Extract user preferences from query and response"""
        preferences = []
        
        # Check for preference indicators in query
        if any(phrase in query.lower() for phrase in [
            'i like', 'i love', 'i prefer', 'i want', 'i need',
            'i don\'t like', 'i hate', 'i dislike'
        ]):
            # Extract the preference from the query
            for phrase in ['i like', 'i love', 'i prefer', 'i want', 'i need']:
                if phrase in query.lower():
                    match = re.search(f"{phrase}\\s+(.+)", query.lower())
                    if match:
                        preferences.append({
                            'type': 'preference',
                            'content': f"User likes {match.group(1)}",
                            'source': 'query'
                        })
            
            for phrase in ['i don\'t like', 'i hate', 'i dislike']:
                if phrase in query.lower():
                    match = re.search(f"{phrase}\\s+(.+)", query.lower())
                    if match:
                        preferences.append({
                            'type': 'preference',
                            'content': f"User dislikes {match.group(1)}",
                            'source': 'query'
                        })
        
        # Check for preference acknowledgments in response
        if any(phrase in response.lower() for phrase in [
            'you like', 'you prefer', 'you want', 'you need',
            'you don\'t like', 'you dislike'
        ]):
            # Extract the preference from the response
            for phrase in ['you like', 'you prefer', 'you want', 'you need']:
                if phrase in response.lower():
                    match = re.search(f"{phrase}\\s+(.+?)[.!?]", response.lower())
                    if match:
                        preferences.append({
                            'type': 'preference',
                            'content': f"User likes {match.group(1)}",
                            'source': 'response'
                        })
            
            for phrase in ['you don\'t like', 'you dislike']:
                if phrase in response.lower():
                    match = re.search(f"{phrase}\\s+(.+?)[.!?]", response.lower())
                    if match:
                        preferences.append({
                            'type': 'preference',
                            'content': f"User dislikes {match.group(1)}",
                            'source': 'response'
                        })
        
        return preferences
    
    def _extract_insights(self, query, response):
        """Extract important insights from the conversation"""
        insights = []
        
        # Look for summary or conclusion indicators
        summary_indicators = [
            'in summary', 'to summarize', 'in conclusion',
            'overall', 'the key point', 'the main idea',
            'importantly', 'significantly', 'notably'
        ]
        
        sentences = re.split(r'[.!?]', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for summary indicators
            if any(indicator in sentence.lower() for indicator in summary_indicators):
                insights.append({
                    'type': 'insight',
                    'content': sentence,
                    'source': 'response'
                })
        
        return insights
    
    def _score_memory_items(self, memory_items):
        """Score memory items based on importance and relevance"""
        scored_items = {
            'facts': [],
            'preferences': [],
            'insights': []
        }
        
        # Score facts
        for fact in memory_items['facts']:
            score = 0.5  # Base score
            
            # Adjust score based on content
            if len(fact['content'].split()) > 5:
                score += 0.1  # Longer facts may be more substantial
            
            if any(entity in fact['content'].lower() for entity in ['i', 'you', 'we', 'our', 'your']):
                score += 0.2  # Personal facts are more relevant
            
            scored_items['facts'].append({
                **fact,
                'score': min(1.0, score)
            })
        
        # Score preferences (preferences are inherently important)
        for pref in memory_items['preferences']:
            score = 0.8  # High base score for preferences
            
            # Adjust score based on source
            if pref['source'] == 'query':
                score += 0.1  # Direct user statements are more reliable
            
            scored_items['preferences'].append({
                **pref,
                'score': min(1.0, score)
            })
        
        # Score insights
        for insight in memory_items['insights']:
            score = 0.6  # Base score
            
            # Adjust score based on content
            if len(insight['content'].split()) > 10:
                score += 0.1  # Longer insights may be more substantial
            
            if any(entity in insight['content'].lower() for entity in ['i', 'you', 'we', 'our', 'your']):
                score += 0.2  # Personal insights are more relevant
            
            scored_items['insights'].append({
                **insight,
                'score': min(1.0, score)
            })
        
        return scored_items
    
    def _filter_memory_worthy(self, scored_items, threshold=0.6):
        """Filter items based on memory-worthiness threshold"""
        memory_worthy = {
            'facts': [item for item in scored_items['facts'] if item['score'] >= threshold],
            'preferences': [item for item in scored_items['preferences'] if item['score'] >= threshold],
            'insights': [item for item in scored_items['insights'] if item['score'] >= threshold]
        }
        
        return memory_worthy
```

### 6. Memory Update

#### Functionality:
- Updates memory systems with new information
- Routes different types of information to appropriate stores
- Handles deduplication and conflict resolution
- Manages memory consolidation and optimization

#### Implementation:
```python
class MemoryUpdater:
    def __init__(self, rag_system, mem0_system):
        """
        Initialize the Memory Updater.
        
        Args:
            rag_system: RAG component instance
            mem0_system: Mem0 component instance
        """
        self.rag_system = rag_system
        self.mem0_system = mem0_system
    
    def update(self, query, response, analysis_results):
        """
        Update memory systems with new information.
        
        Args:
            query: The user's query text
            response: The LLM's response text
            analysis_results: Results from ResponseAnalyzer
            
        Returns:
            dict: Update statistics
        """
        memory_worthy = analysis_results['memory_worthy']
        
        # Update statistics
        stats = {
            'rag_updates': 0,
            'mem0_updates': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # Update RAG with facts and insights
        rag_items = memory_worthy['facts'] + memory_worthy['insights']
        for item in rag_items:
            try:
                # Check for duplicates before adding
                if not self.rag_system.contains_similar(item['content']):
                    self.rag_system.add_memory(
                        content=item['content'],
                        metadata={
                            'type': item['type'],
                            'source': item['source'],
                            'timestamp': datetime.now().isoformat(),
                            'query': query,
                            'response': response
                        }
                    )
                    stats['rag_updates'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                print(f"Error updating RAG: {str(e)}")
                stats['errors'] += 1
        
        # Update Mem0 with preferences
        for item in memory_worthy['preferences']:
            try:
                # Check for conflicts before adding
                if not self.mem0_system.has_conflicting_preference(item['content']):
                    self.mem0_system.add_preference(
                        content=item['content'],
                        metadata={
                            'source': item['source'],
                            'timestamp': datetime.now().isoformat(),
                            'confidence': item['score']
                        }
                    )
                    stats['mem0_updates'] += 1
                else:
                    # Handle conflict (e.g., update existing preference)
                    self.mem0_system.update_preference(
                        content=item['content'],
                        metadata={
                            'source': item['source'],
                            'timestamp': datetime.now().isoformat(),
                            'confidence': item['score']
                        }
                    )
                    stats['mem0_updates'] += 1
            except Exception as e:
                print(f"Error updating Mem0: {str(e)}")
                stats['errors'] += 1
        
        # Periodically trigger memory consolidation
        if random.random() < 0.1:  # 10% chance after each update
            self._consolidate_memories()
        
        return stats
    
    def _consolidate_memories(self):
        """Periodically consolidate and optimize memory stores"""
        try:
            # Consolidate RAG memories
            self.rag_system.consolidate()
            
            # Consolidate Mem0 memories
            self.mem0_system.consolidate()
            
            print("Memory consolidation completed")
        except Exception as e:
            print(f"Error during memory consolidation: {str(e)}")
```

## System Workflow

### Query Processing Flow

```
User Query → Query Analyzer → Memory Retrieval → Context Optimization → Prompt Assembly → LLM API
```

1. **User Query**: The user submits a question or statement
2. **Query Analyzer**: Determines query type and memory needs
3. **Memory Retrieval**: Fetches relevant information from RAG and Mem0
4. **Context Optimization**: Optimizes retrieved context to fit token limits
5. **Prompt Assembly**: Creates the final prompt with optimized context
6. **LLM API**: Sends the prompt to the LLM for processing

### Response Processing Flow

```
LLM Response → Response Analyzer → Memory Update → Memory Consolidation
```

1. **LLM Response**: The LLM generates a response based on the prompt
2. **Response Analyzer**: Identifies memory-worthy information in the response
3. **Memory Update**: Stores new information in appropriate memory systems
4. **Memory Consolidation**: Periodically optimizes and consolidates memory

## System Instructions Template

The system uses a template for LLM instructions that explains how to use the provided memory context:

```
You are an AI assistant with access to two memory systems:

1. RAG: A knowledge base containing factual information from past conversations
2. Mem0: A personalized memory containing user preferences and context

When responding:
- Use the provided memory context to inform your responses
- Maintain awareness of user preferences stored in Mem0
- Be consistent with past interactions referenced in the memory
- When appropriate, acknowledge relevant past conversations

{memory_context}

Remember that you should respond naturally without explicitly mentioning the memory systems. Simply use the information to provide more personalized and contextually relevant responses.
```

## Token Budget Management

### Token Allocation Strategy

The system dynamically allocates tokens between different components:

1. **System Instructions**: Fixed allocation (typically 150-300 tokens)
2. **Memory Context**: Dynamic allocation based on query type (typically 500-1500 tokens)
   - RAG: 30-70% of memory context tokens
   - Mem0: 30-70% of memory context tokens
3. **Conversation History**: Remaining tokens after other allocations
4. **User Query**: Always included in full

### Optimization Techniques

1. **Truncation**: Removing less relevant content when token limits are reached
2. **Compression**: Summarizing or condensing information to reduce token usage
3. **Prioritization**: Allocating more tokens to more relevant memory types
4. **Dynamic Adjustment**: Adjusting allocations based on query characteristics

## Implementation Considerations

### Performance Optimization

- **Caching**: Cache analysis results and retrieved memories for similar queries
- **Asynchronous Updates**: Perform memory updates asynchronously after response delivery
- **Batch Processing**: Consolidate memory in batches during idle periods
- **Incremental Analysis**: Analyze only new content in ongoing conversations

### Error Handling

- **Retrieval Fallbacks**: Default to conversation history if memory retrieval fails
- **Graceful Degradation**: Continue functioning with reduced capabilities if components fail
- **Retry Logic**: Implement retries for transient errors in memory operations
- **Logging**: Comprehensive logging for debugging and monitoring

### Scalability

- **Modular Design**: Components can be scaled independently
- **Stateless Operation**: Core components operate statelessly for horizontal scaling
- **Efficient Storage**: Optimize memory storage for growing data volumes
- **Pruning Strategies**: Automatically remove or archive less useful memories

## Next Steps

1. Implement the Query Analyzer component
2. Develop the Memory Retrieval integration
3. Create the Context Optimization logic
4. Build the Prompt Assembly system
5. Implement the Response Analyzer
6. Develop the Memory Update mechanisms
7. Test with various query types and conversation scenarios
8. Optimize token usage and retrieval relevance
