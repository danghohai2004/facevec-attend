"use client";

import * as React from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { deleteEmployee, getEmployeesByName, listEmployees } from "@/lib/api";
import type { Employee } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const PAGE_SIZES = ["10", "20", "50"];

export function EmployeeList() {
  const [page, setPage] = React.useState(1);
  const [pageSize, setPageSize] = React.useState(10);
  const [search, setSearch] = React.useState("");
  const [selectedEmployee, setSelectedEmployee] = React.useState<Employee | null>(
    null,
  );

  const queryClient = useQueryClient();
  const searchTerm = search.trim();

  const listQuery = useQuery({
    queryKey: ["employees", page, pageSize],
    queryFn: () => listEmployees({ page, pageSize }),
    enabled: !searchTerm,
  });
  const nameQuery = useQuery({
    queryKey: ["employee-name", searchTerm],
    queryFn: () => getEmployeesByName(searchTerm),
    enabled: Boolean(searchTerm),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteEmployee,
    onSuccess: () => {
      toast.success("Employee removed successfully.");
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      queryClient.invalidateQueries({ queryKey: ["employee-name"] });
      setSelectedEmployee(null);
    },
    onError: (error: Error) => {
      toast.error(error.message || "Unable to delete employee.");
    },
  });

  const items = searchTerm ? nameQuery.data ?? [] : listQuery.data?.items ?? [];

  const total = searchTerm
    ? nameQuery.data?.length ?? 0
    : listQuery.data?.total ?? 0;

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const currentPage = searchTerm ? 1 : page;
  const isLoading = searchTerm ? nameQuery.isLoading : listQuery.isLoading;
  const isError = searchTerm ? nameQuery.isError : listQuery.isError;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Input
          placeholder="Search by employee name"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="max-w-xs"
        />
        <div className="flex items-center gap-3">
          <Select
            value={`${pageSize}`}
            onValueChange={(value) => {
              setPageSize(Number(value));
              setPage(1);
            }}
          >
            <SelectTrigger className="w-[120px]">
              <SelectValue placeholder="Page size" />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZES.map((size) => (
                <SelectItem key={size} value={size}>
                  {size} / page
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Badge variant="outline">
            {total} employees
          </Badge>
        </div>
      </div>

      <div className="rounded-[3px] border-2 border-foreground shadow-brutal-sm">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Employee Code</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  Loading employees...
                </TableCell>
              </TableRow>
            )}
            {isError && (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  Unable to load employee data.
                </TableCell>
              </TableRow>
            )}
            {!isLoading && items.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  No employees found.
                </TableCell>
              </TableRow>
            )}
            {items.map((employee) => (
              <TableRow key={employee.id}>
                <TableCell className="font-medium">{employee.name}</TableCell>
                <TableCell>{employee.emp_code}</TableCell>
                <TableCell>
                  <Badge variant="secondary">
                    {employee.status ?? "Active"}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSelectedEmployee(employee)}
                  >
                    Delete
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs font-bold text-muted-foreground">
          Page {currentPage} of {totalPages}
        </p>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((prev) => Math.max(1, prev - 1))}
            disabled={searchTerm !== "" || page <= 1}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
            disabled={searchTerm !== "" || page >= totalPages}
          >
            Next
          </Button>
        </div>
      </div>

      <Dialog open={!!selectedEmployee} onOpenChange={() => setSelectedEmployee(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove employee</DialogTitle>
            <DialogDescription>
              This will permanently remove the employee and embeddings.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSelectedEmployee(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() =>
                selectedEmployee && deleteMutation.mutate(selectedEmployee.id)
              }
              disabled={deleteMutation.isPending}
            >
              Confirm Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
