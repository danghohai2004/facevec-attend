"use client";

import { EmployeeList } from "@/components/employees/employee-list";
import { EmployeeRegistration } from "@/components/employees/employee-registration";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function EmployeesClient() {
  return (
    <Tabs defaultValue="list" className="space-y-6">
      <TabsList>
        <TabsTrigger value="list">Employee List</TabsTrigger>
        <TabsTrigger value="register">Register Employee</TabsTrigger>
      </TabsList>
      <TabsContent value="list">
        <EmployeeList />
      </TabsContent>
      <TabsContent value="register">
        <EmployeeRegistration />
      </TabsContent>
    </Tabs>
  );
}
