"use client";

import { useEffect, useState } from "react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, X } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

type TrainingStatus = "idle" | "queued" | "training" | "completed";

export default function UserTrainingPage() {
  const [file, setFile] = useState<File | null>(null);

  const [status, setStatus] = useState<TrainingStatus>("idle");

  const [progress, setProgress] = useState(0);

  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (message: string) => {
    setLogs((prev) => [
      ...prev,
      `${new Date().toLocaleTimeString()} - ${message}`,
    ]);
  };

  const handleTrain = () => {
    if (!file) return;

    setLogs([]);
    setProgress(0);

    setStatus("queued");

    addLog("Training job created");
    addLog(`Dataset loaded: ${file.name}`);

    setTimeout(() => {
      setStatus("training");
      addLog("Initializing training pipeline");
    }, 1000);
  };

  useEffect(() => {
    if (status !== "training") return;

    const steps = [
      "Validating dataset",
      "Preprocessing records",
      "Generating embeddings",
      "Training model",
      "Evaluating model",
      "Saving checkpoint",
      "Finalizing model",
    ];

    let current = 0;

    const interval = setInterval(() => {
      current += 10;

      setProgress(current);

      if (current % 20 === 0) {
        const idx = current / 20 - 1;

        if (steps[idx]) {
          addLog(steps[idx]);
        }
      }

      if (current >= 100) {
        clearInterval(interval);

        setProgress(100);

        setStatus("completed");

        addLog("Training completed successfully");
      }
    }, 700);

    return () => clearInterval(interval);
  }, [status]);

  const history = [
    {
      id: "001",
      dataset: "stories.csv",
      status: "Completed",
      date: "2026-06-15",
    },
    {
      id: "002",
      dataset: "blogs.docx",
      status: "Completed",
      date: "2026-06-17",
    },
    {
      id: "003",
      dataset: "scripts.pdf",
      status: "Completed",
      date: "2026-06-19",
    },
  ];

  return (
    <div className="mx-auto min-w-6xl p-8 space-y-6">
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Training Configuration</CardTitle>

            <CardDescription>Dataset selection / upload</CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="space-y-4">
              {!file ? (
                <label
                  htmlFor="dataset-upload"
                  className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 text-center transition-colors hover:bg-muted/50"
                >
                  <Upload className="mb-3 h-10 w-10 text-muted-foreground" />

                  <h3 className="font-medium">Upload Dataset</h3>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Text files only
                  </p>

                  <input
                    id="dataset-upload"
                    type="file"
                    accept=".txt,.doc,.docx,.pdf"
                    className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                </label>
              ) : (
                <div className="rounded-xl border bg-muted/40 p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <FileText className="h-8 w-8 text-primary" />

                      <div>
                        <p className="font-medium">{file.name}</p>

                        <p className="text-sm text-muted-foreground">
                          {(file.size / 1024).toFixed(2)} KB
                        </p>
                      </div>
                    </div>

                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => setFile(null)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </div>

            <Separator />

            <Button
              className="w-full"
              disabled={!file || status === "training"}
              onClick={handleTrain}
            >
              Train
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>

            <CardDescription>Progress Tracking</CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <Badge>{status.toUpperCase()}</Badge>

            <Progress value={progress} />

            <p className="text-sm text-muted-foreground">
              {progress}% Complete
            </p>

            <ScrollArea className="h-64 rounded-md border p-3">
              <div className="space-y-2 text-sm font-mono">
                {logs.length === 0 ? (
                  <p className="text-muted-foreground">No logs available</p>
                ) : (
                  logs.map((log, idx) => <p key={idx}>{log}</p>)
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Training History</CardTitle>

          <CardDescription>Previously trained models</CardDescription>
        </CardHeader>

        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job ID</TableHead>
                <TableHead>Dataset</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {history.map((job) => (
                <TableRow key={job.id}>
                  <TableCell>{job.id}</TableCell>
                  <TableCell>{job.dataset}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{job.status}</Badge>
                  </TableCell>
                  <TableCell>{job.date}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
