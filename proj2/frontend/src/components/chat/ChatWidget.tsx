import { useState, useEffect, useRef } from 'react';
import { MessageSquare, X, Send, Minimize2, Maximize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { chatApi, type ChatMessageResponse } from '@/lib/api';
import { cn } from '@/lib/utils';

export function ChatWidget() {
    const [isOpen, setIsOpen] = useState(false);
    const [isMinimized, setIsMinimized] = useState(false);
    const [messages, setMessages] = useState<ChatMessageResponse[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Load session on open
    useEffect(() => {
        if (isOpen && !sessionId) {
            loadSession();
        }
    }, [isOpen]);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, isOpen]);

    const loadSession = async () => {
        try {
            const sessions = await chatApi.getSessions();
            if (sessions.length > 0) {
                // Load most recent session
                const recentSession = sessions[0];
                setSessionId(recentSession.id);
                // Fetch full history
                const fullSession = await chatApi.getSession(recentSession.id);
                setMessages(fullSession.messages);
            }
        } catch (error) {
            console.error('Failed to load chat session:', error);
        }
    };

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: ChatMessageResponse = {
            id: 'temp-' + Date.now(),
            role: 'user',
            content: input,
            created_at: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await chatApi.sendMessage({
                message: userMsg.content,
                session_id: sessionId || undefined,
            });

            setSessionId(response.session_id);

            const aiMsg: ChatMessageResponse = {
                id: 'ai-' + Date.now(),
                role: 'model',
                content: response.response,
                created_at: new Date().toISOString(),
            };

            setMessages((prev) => [...prev, aiMsg]);
        } catch (error) {
            console.error('Failed to send message:', error);
            // Add error message
            setMessages((prev) => [
                ...prev,
                {
                    id: 'error-' + Date.now(),
                    role: 'system',
                    content: `Failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
                    created_at: new Date().toISOString(),
                },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) {
        return (
            <Button
                className="fixed bottom-4 right-4 h-14 w-14 rounded-full shadow-lg z-50"
                onClick={() => setIsOpen(true)}
            >
                <MessageSquare className="h-6 w-6" />
            </Button>
        );
    }

    return (
        <Card className={cn(
            "fixed bottom-4 right-4 w-[350px] shadow-xl z-50 transition-all duration-300",
            isMinimized ? "h-[60px]" : "h-[500px]"
        )}>
            <CardHeader className="p-4 border-b flex flex-row items-center justify-between space-y-0">
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    AI Health Concierge
                </CardTitle>
                <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setIsMinimized(!isMinimized)}>
                        {isMinimized ? <Maximize2 className="h-3 w-3" /> : <Minimize2 className="h-3 w-3" />}
                    </Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setIsOpen(false)}>
                        <X className="h-3 w-3" />
                    </Button>
                </div>
            </CardHeader>

            {!isMinimized && (
                <>
                    <CardContent className="p-0 flex-1 h-[380px] overflow-hidden">
                        <div className="h-full p-4 overflow-y-auto">
                            <div className="space-y-4">
                                {messages.length === 0 && (
                                    <div className="text-center text-muted-foreground text-sm py-8">
                                        <p>Hi! I'm your personal health assistant.</p>
                                        <p>Ask me anything about your nutrition or goals.</p>
                                    </div>
                                )}
                                {messages.map((msg) => (
                                    <div
                                        key={msg.id}
                                        className={cn(
                                            "flex w-full",
                                            msg.role === 'user' ? "justify-end" : "justify-start"
                                        )}
                                    >
                                        <div
                                            className={cn(
                                                "rounded-lg px-3 py-2 max-w-[80%] text-sm",
                                                msg.role === 'user'
                                                    ? "bg-primary text-primary-foreground"
                                                    : msg.role === 'system'
                                                        ? "bg-destructive/10 text-destructive"
                                                        : "bg-muted"
                                            )}
                                        >
                                            {msg.content}
                                        </div>
                                    </div>
                                ))}
                                {isLoading && (
                                    <div className="flex justify-start">
                                        <div className="bg-muted rounded-lg px-3 py-2 text-sm animate-pulse">
                                            Thinking...
                                        </div>
                                    </div>
                                )}
                                <div ref={scrollRef} />
                            </div>
                        </div>
                    </CardContent>
                    <CardFooter className="p-3 border-t">
                        <form
                            onSubmit={(e) => {
                                e.preventDefault();
                                handleSend();
                            }}
                            className="flex w-full gap-2"
                        >
                            <Input
                                placeholder="Type a message..."
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                disabled={isLoading}
                            />
                            <Button type="submit" size="icon" disabled={isLoading || !input.trim()}>
                                <Send className="h-4 w-4" />
                            </Button>
                        </form>
                    </CardFooter>
                </>
            )}
        </Card>
    );
}
