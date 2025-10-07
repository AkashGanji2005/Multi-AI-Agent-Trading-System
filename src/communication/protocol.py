"""
Communication Protocol for Multi-Agent Marketplace
Handles message passing, negotiation, and information sharing between agents
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import uuid

from ..environment.marketplace import Message, MessageType, AgentType

class CommunicationProtocol(Enum):
    DIRECT = "direct"           # Direct agent-to-agent communication
    BROADCAST = "broadcast"     # One-to-many broadcasting
    MULTICAST = "multicast"     # Group communication
    GOSSIP = "gossip"          # Gossip protocol for information spread
    AUCTION = "auction"        # Auction-style communication
    NEGOTIATION = "negotiation" # Structured negotiation protocol

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class ChannelType(Enum):
    PUBLIC = "public"           # Open to all agents
    PRIVATE = "private"         # Between specific agents
    GROUP = "group"            # Group/alliance channel
    REGULATORY = "regulatory"   # Regulatory announcements
    EMERGENCY = "emergency"     # Emergency communications

@dataclass
class CommunicationChannel:
    """Communication channel for message routing"""
    channel_id: str
    channel_type: ChannelType
    participants: Set[str]
    moderator: Optional[str] = None
    max_participants: int = 100
    message_history: List[Message] = None
    is_encrypted: bool = False
    bandwidth_limit: int = 1000  # Messages per time unit
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.message_history is None:
            self.message_history = []
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class NegotiationSession:
    """Structured negotiation session between agents"""
    session_id: str
    participants: List[str]
    mediator: Optional[str] = None
    topic: str = ""
    status: str = "active"  # active, paused, completed, failed
    rounds: List[Dict[str, Any]] = None
    current_round: int = 0
    max_rounds: int = 10
    timeout: float = 300.0  # 5 minutes default
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.rounds is None:
            self.rounds = []
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class AuctionSession:
    """Auction session for competitive bidding"""
    auction_id: str
    auctioneer: str
    item_description: Dict[str, Any]
    auction_type: str = "english"  # english, dutch, sealed_bid, vickrey
    participants: Set[str] = None
    bids: List[Dict[str, Any]] = None
    status: str = "open"  # open, closed, completed
    start_time: float = 0.0
    end_time: float = 0.0
    reserve_price: float = 0.0
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = set()
        if self.bids is None:
            self.bids = []
        if self.start_time == 0.0:
            self.start_time = time.time()

class CommunicationManager:
    """
    Central communication manager that handles all inter-agent communication
    Supports multiple protocols, channels, and advanced features
    """
    
    def __init__(self):
        # Core communication infrastructure
        self.channels: Dict[str, CommunicationChannel] = {}
        self.message_queue: deque = deque()
        self.agent_inboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_outboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Protocol-specific managers
        self.negotiation_sessions: Dict[str, NegotiationSession] = {}
        self.auction_sessions: Dict[str, AuctionSession] = {}
        
        # Network topology and routing
        self.network_topology: Dict[str, Set[str]] = defaultdict(set)
        self.routing_table: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self.agent_locations: Dict[str, Dict[str, float]] = {}  # For geographic routing
        
        # Message filtering and processing
        self.message_filters: Dict[str, List[Callable]] = defaultdict(list)
        self.message_processors: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.spam_detection: Dict[str, int] = defaultdict(int)
        
        # Performance and monitoring
        self.message_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.latency_stats: Dict[str, List[float]] = defaultdict(list)
        self.bandwidth_usage: Dict[str, int] = defaultdict(int)
        
        # Security and trust
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.reputation_system: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.5))
        self.blocked_agents: Dict[str, Set[str]] = defaultdict(set)
        
        # Create default channels
        self._create_default_channels()
        
        # Background processing
        self.processing_thread = None
        self.is_running = False
    
    def _create_default_channels(self):
        """Create default communication channels"""
        
        # Public announcement channel
        self.create_channel(
            "public_announcements",
            ChannelType.PUBLIC,
            participants=set(),
            max_participants=1000
        )
        
        # Regulatory channel
        self.create_channel(
            "regulatory_announcements",
            ChannelType.REGULATORY,
            participants=set(),
            max_participants=1000
        )
        
        # Emergency channel
        self.create_channel(
            "emergency_communications",
            ChannelType.EMERGENCY,
            participants=set(),
            max_participants=1000
        )
        
        # Trading floor (public trading discussions)
        self.create_channel(
            "trading_floor",
            ChannelType.PUBLIC,
            participants=set(),
            max_participants=500
        )
    
    def start(self):
        """Start the communication manager"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_messages_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop(self):
        """Stop the communication manager"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
    
    def register_agent(self, agent_id: str, agent_type: AgentType, location: Dict[str, float] = None):
        """Register an agent with the communication system"""
        
        # Add to network topology
        self.network_topology[agent_id] = set()
        
        # Set location (for geographic routing if needed)
        if location:
            self.agent_locations[agent_id] = location
        else:
            self.agent_locations[agent_id] = {"x": 0.0, "y": 0.0}
        
        # Add to relevant default channels
        if agent_type == AgentType.REGULATOR:
            self.join_channel(agent_id, "regulatory_announcements")
            self.channels["regulatory_announcements"].moderator = agent_id
        
        self.join_channel(agent_id, "public_announcements")
        self.join_channel(agent_id, "trading_floor")
        self.join_channel(agent_id, "emergency_communications")
        
        # Initialize statistics
        self.message_stats[agent_id] = defaultdict(int)
    
    def create_channel(self, 
                      channel_id: str, 
                      channel_type: ChannelType,
                      participants: Set[str] = None,
                      moderator: str = None,
                      **kwargs) -> bool:
        """Create a new communication channel"""
        
        if channel_id in self.channels:
            return False
        
        self.channels[channel_id] = CommunicationChannel(
            channel_id=channel_id,
            channel_type=channel_type,
            participants=participants or set(),
            moderator=moderator,
            **kwargs
        )
        
        return True
    
    def join_channel(self, agent_id: str, channel_id: str) -> bool:
        """Add an agent to a communication channel"""
        
        if channel_id not in self.channels:
            return False
        
        channel = self.channels[channel_id]
        
        if len(channel.participants) >= channel.max_participants:
            return False
        
        channel.participants.add(agent_id)
        return True
    
    def leave_channel(self, agent_id: str, channel_id: str) -> bool:
        """Remove an agent from a communication channel"""
        
        if channel_id not in self.channels:
            return False
        
        channel = self.channels[channel_id]
        channel.participants.discard(agent_id)
        return True
    
    def send_message(self, 
                    sender: str,
                    message: Message,
                    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
                    priority: MessagePriority = MessagePriority.NORMAL,
                    channel: str = None) -> bool:
        """Send a message using specified protocol"""
        
        # Apply spam detection
        if self._is_spam(sender, message):
            return False
        
        # Apply message filters
        if not self._apply_filters(sender, message):
            return False
        
        # Route message based on protocol
        if protocol == CommunicationProtocol.DIRECT:
            return self._send_direct_message(sender, message, priority)
        elif protocol == CommunicationProtocol.BROADCAST:
            return self._send_broadcast_message(sender, message, channel, priority)
        elif protocol == CommunicationProtocol.MULTICAST:
            return self._send_multicast_message(sender, message, channel, priority)
        elif protocol == CommunicationProtocol.GOSSIP:
            return self._send_gossip_message(sender, message, priority)
        else:
            return self._send_direct_message(sender, message, priority)
    
    def _send_direct_message(self, sender: str, message: Message, priority: MessagePriority) -> bool:
        """Send direct message to specific recipient"""
        
        if message.receiver not in self.agent_inboxes:
            return False
        
        # Check if sender is blocked by receiver
        if sender in self.blocked_agents[message.receiver]:
            return False
        
        # Add to recipient's inbox with priority
        prioritized_message = {
            'message': message,
            'priority': priority.value,
            'timestamp': time.time(),
            'sender': sender
        }
        
        self.agent_inboxes[message.receiver].append(prioritized_message)
        
        # Update statistics
        self.message_stats[sender]['sent'] += 1
        self.message_stats[message.receiver]['received'] += 1
        
        return True
    
    def _send_broadcast_message(self, sender: str, message: Message, channel: str, priority: MessagePriority) -> bool:
        """Broadcast message to all agents in a channel"""
        
        if channel and channel in self.channels:
            recipients = self.channels[channel].participants
        else:
            recipients = set(self.agent_inboxes.keys())
        
        success_count = 0
        
        for recipient in recipients:
            if recipient != sender and recipient not in self.blocked_agents[sender]:
                broadcast_message = Message(
                    sender=sender,
                    receiver=recipient,
                    message_type=message.message_type,
                    content=message.content,
                    timestamp=message.timestamp,
                    response_to=message.response_to
                )
                
                if self._send_direct_message(sender, broadcast_message, priority):
                    success_count += 1
        
        return success_count > 0
    
    def _send_multicast_message(self, sender: str, message: Message, channel: str, priority: MessagePriority) -> bool:
        """Send message to specific group of agents"""
        
        if not channel or channel not in self.channels:
            return False
        
        channel_obj = self.channels[channel]
        if sender not in channel_obj.participants:
            return False
        
        return self._send_broadcast_message(sender, message, channel, priority)
    
    def _send_gossip_message(self, sender: str, message: Message, priority: MessagePriority) -> bool:
        """Send message using gossip protocol for information spread"""
        
        # Select random subset of connected agents
        connected_agents = list(self.network_topology[sender])
        if not connected_agents:
            connected_agents = list(self.agent_inboxes.keys())
        
        # Gossip to random subset (typically 3-5 agents)
        gossip_count = min(5, len(connected_agents))
        import random
        gossip_targets = random.sample(connected_agents, gossip_count)
        
        success_count = 0
        for target in gossip_targets:
            if target != sender:
                gossip_message = Message(
                    sender=sender,
                    receiver=target,
                    message_type=message.message_type,
                    content={**message.content, 'gossip': True, 'hop_count': message.content.get('hop_count', 0) + 1},
                    timestamp=message.timestamp,
                    response_to=message.response_to
                )
                
                if self._send_direct_message(sender, gossip_message, priority):
                    success_count += 1
        
        return success_count > 0
    
    def receive_messages(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """Retrieve messages for an agent"""
        
        messages = []
        inbox = self.agent_inboxes[agent_id]
        
        # Sort by priority and timestamp
        sorted_messages = sorted(list(inbox), key=lambda x: (-x['priority'], x['timestamp']))
        
        for _ in range(min(max_messages, len(sorted_messages))):
            if sorted_messages:
                msg_data = sorted_messages.pop(0)
                messages.append(msg_data['message'])
                inbox.remove(msg_data)
        
        return messages
    
    def start_negotiation(self, 
                         initiator: str,
                         participants: List[str],
                         topic: str,
                         mediator: str = None,
                         max_rounds: int = 10) -> str:
        """Start a structured negotiation session"""
        
        session_id = f"negotiation_{uuid.uuid4().hex[:8]}"
        
        session = NegotiationSession(
            session_id=session_id,
            participants=participants,
            mediator=mediator,
            topic=topic,
            max_rounds=max_rounds
        )
        
        self.negotiation_sessions[session_id] = session
        
        # Notify participants
        for participant in participants:
            notification = Message(
                sender="system",
                receiver=participant,
                message_type=MessageType.MEDIATION_REQUEST,
                content={
                    'session_id': session_id,
                    'topic': topic,
                    'participants': participants,
                    'mediator': mediator,
                    'max_rounds': max_rounds
                },
                timestamp=int(time.time())
            )
            
            self._send_direct_message("system", notification, MessagePriority.HIGH)
        
        return session_id
    
    def add_negotiation_round(self, 
                             session_id: str,
                             participant: str,
                             proposal: Dict[str, Any]) -> bool:
        """Add a proposal round to negotiation session"""
        
        if session_id not in self.negotiation_sessions:
            return False
        
        session = self.negotiation_sessions[session_id]
        
        if participant not in session.participants or session.status != "active":
            return False
        
        round_data = {
            'round_number': session.current_round,
            'participant': participant,
            'proposal': proposal,
            'timestamp': time.time()
        }
        
        session.rounds.append(round_data)
        
        # Notify other participants
        for other_participant in session.participants:
            if other_participant != participant:
                notification = Message(
                    sender="system",
                    receiver=other_participant,
                    message_type=MessageType.OFFER,
                    content={
                        'session_id': session_id,
                        'round': session.current_round,
                        'proposer': participant,
                        'proposal': proposal
                    },
                    timestamp=int(time.time())
                )
                
                self._send_direct_message("system", notification, MessagePriority.HIGH)
        
        session.current_round += 1
        
        # Check if negotiation should end
        if session.current_round >= session.max_rounds:
            session.status = "completed"
        
        return True
    
    def start_auction(self,
                     auctioneer: str,
                     item_description: Dict[str, Any],
                     auction_type: str = "english",
                     duration: float = 300.0,
                     reserve_price: float = 0.0) -> str:
        """Start an auction session"""
        
        auction_id = f"auction_{uuid.uuid4().hex[:8]}"
        
        auction = AuctionSession(
            auction_id=auction_id,
            auctioneer=auctioneer,
            item_description=item_description,
            auction_type=auction_type,
            end_time=time.time() + duration,
            reserve_price=reserve_price
        )
        
        self.auction_sessions[auction_id] = auction
        
        # Broadcast auction announcement
        announcement = Message(
            sender=auctioneer,
            receiver="all",
            message_type=MessageType.OFFER,
            content={
                'auction_id': auction_id,
                'item_description': item_description,
                'auction_type': auction_type,
                'duration': duration,
                'reserve_price': reserve_price,
                'start_time': auction.start_time
            },
            timestamp=int(time.time())
        )
        
        self._send_broadcast_message(auctioneer, announcement, "trading_floor", MessagePriority.HIGH)
        
        return auction_id
    
    def place_bid(self, auction_id: str, bidder: str, bid_amount: float, bid_data: Dict[str, Any] = None) -> bool:
        """Place a bid in an auction"""
        
        if auction_id not in self.auction_sessions:
            return False
        
        auction = self.auction_sessions[auction_id]
        
        if auction.status != "open" or time.time() > auction.end_time:
            return False
        
        if bid_amount < auction.reserve_price:
            return False
        
        bid = {
            'bidder': bidder,
            'amount': bid_amount,
            'timestamp': time.time(),
            'data': bid_data or {}
        }
        
        auction.bids.append(bid)
        auction.participants.add(bidder)
        
        # Notify auctioneer and other participants (if appropriate for auction type)
        if auction.auction_type == "english":
            # In English auctions, all participants see current highest bid
            current_high_bid = max(auction.bids, key=lambda x: x['amount'])
            
            notification = Message(
                sender="system",
                receiver="all",
                message_type=MessageType.INFORMATION_SHARE,
                content={
                    'auction_id': auction_id,
                    'new_bid': bid_amount,
                    'current_high_bid': current_high_bid['amount'],
                    'bidder_count': len(auction.participants)
                },
                timestamp=int(time.time())
            )
            
            # Send to all auction participants
            for participant in auction.participants:
                notification.receiver = participant
                self._send_direct_message("system", notification, MessagePriority.HIGH)
        
        return True
    
    def _is_spam(self, sender: str, message: Message) -> bool:
        """Simple spam detection"""
        
        current_time = time.time()
        self.spam_detection[sender] += 1
        
        # Reset counter every minute
        if hasattr(self, '_last_spam_reset'):
            if current_time - self._last_spam_reset > 60:
                self.spam_detection.clear()
                self._last_spam_reset = current_time
        else:
            self._last_spam_reset = current_time
        
        # More than 50 messages per minute is considered spam
        return self.spam_detection[sender] > 50
    
    def _apply_filters(self, sender: str, message: Message) -> bool:
        """Apply message filters"""
        
        filters = self.message_filters.get(message.receiver, [])
        
        for filter_func in filters:
            if not filter_func(sender, message):
                return False
        
        return True
    
    def _process_messages_loop(self):
        """Background message processing loop"""
        
        while self.is_running:
            try:
                # Process any queued messages
                self._process_message_queue()
                
                # Update auction statuses
                self._update_auctions()
                
                # Update negotiation statuses
                self._update_negotiations()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(0.1)  # 100ms processing interval
                
            except Exception as e:
                print(f"Error in communication processing loop: {e}")
    
    def _process_message_queue(self):
        """Process messages in the queue"""
        
        while self.message_queue:
            try:
                message_data = self.message_queue.popleft()
                # Process message (apply any transformations, logging, etc.)
                
            except IndexError:
                break
    
    def _update_auctions(self):
        """Update auction statuses"""
        
        current_time = time.time()
        
        for auction_id, auction in list(self.auction_sessions.items()):
            if auction.status == "open" and current_time > auction.end_time:
                auction.status = "closed"
                self._close_auction(auction_id)
    
    def _close_auction(self, auction_id: str):
        """Close an auction and notify participants"""
        
        auction = self.auction_sessions[auction_id]
        
        if auction.bids:
            # Determine winner based on auction type
            if auction.auction_type in ["english", "sealed_bid"]:
                winning_bid = max(auction.bids, key=lambda x: x['amount'])
            elif auction.auction_type == "dutch":
                winning_bid = auction.bids[0] if auction.bids else None
            else:
                winning_bid = max(auction.bids, key=lambda x: x['amount'])
            
            if winning_bid:
                # Notify winner
                winner_notification = Message(
                    sender="system",
                    receiver=winning_bid['bidder'],
                    message_type=MessageType.ACCEPT,
                    content={
                        'auction_id': auction_id,
                        'status': 'won',
                        'winning_bid': winning_bid['amount'],
                        'item_description': auction.item_description
                    },
                    timestamp=int(time.time())
                )
                
                self._send_direct_message("system", winner_notification, MessagePriority.URGENT)
                
                # Notify auctioneer
                auctioneer_notification = Message(
                    sender="system",
                    receiver=auction.auctioneer,
                    message_type=MessageType.ACCEPT,
                    content={
                        'auction_id': auction_id,
                        'status': 'sold',
                        'winner': winning_bid['bidder'],
                        'final_price': winning_bid['amount'],
                        'item_description': auction.item_description
                    },
                    timestamp=int(time.time())
                )
                
                self._send_direct_message("system", auctioneer_notification, MessagePriority.URGENT)
        
        auction.status = "completed"
    
    def _update_negotiations(self):
        """Update negotiation session statuses"""
        
        current_time = time.time()
        
        for session_id, session in list(self.negotiation_sessions.items()):
            if session.status == "active" and current_time - session.created_at > session.timeout:
                session.status = "timeout"
                # Notify participants of timeout
                for participant in session.participants:
                    timeout_notification = Message(
                        sender="system",
                        receiver=participant,
                        message_type=MessageType.REJECT,
                        content={
                            'session_id': session_id,
                            'status': 'timeout',
                            'reason': 'negotiation_timeout'
                        },
                        timestamp=int(time.time())
                    )
                    
                    self._send_direct_message("system", timeout_notification, MessagePriority.HIGH)
    
    def _cleanup_old_data(self):
        """Clean up old communication data"""
        
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        # Clean up completed auctions
        for auction_id in list(self.auction_sessions.keys()):
            auction = self.auction_sessions[auction_id]
            if auction.status == "completed" and current_time - auction.end_time > cleanup_age:
                del self.auction_sessions[auction_id]
        
        # Clean up completed negotiations
        for session_id in list(self.negotiation_sessions.keys()):
            session = self.negotiation_sessions[session_id]
            if session.status in ["completed", "failed", "timeout"] and current_time - session.created_at > cleanup_age:
                del self.negotiation_sessions[session_id]
        
        # Clean up old message statistics
        for agent_id in list(self.latency_stats.keys()):
            if len(self.latency_stats[agent_id]) > 1000:
                self.latency_stats[agent_id] = self.latency_stats[agent_id][-500:]
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        
        return {
            'total_channels': len(self.channels),
            'active_agents': len(self.agent_inboxes),
            'total_messages_queued': sum(len(inbox) for inbox in self.agent_inboxes.values()),
            'active_negotiations': len([s for s in self.negotiation_sessions.values() if s.status == "active"]),
            'active_auctions': len([a for a in self.auction_sessions.values() if a.status == "open"]),
            'message_stats': dict(self.message_stats),
            'spam_detections': sum(self.spam_detection.values()),
            'blocked_relationships': sum(len(blocked) for blocked in self.blocked_agents.values())
        }